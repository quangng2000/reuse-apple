#!/usr/bin/env python3
"""Activation-policy tests using tiny synthetic packages, never model evidence."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import package_bundle as target


class PackageActivationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.generated = self.root/'research'
        self.generated.mkdir()
        self.repo = self.root/'repo'
        self.repo.mkdir()
        self.host = self.root/'libreuse_streaming_vulkan.dylib'
        self.android = self.root/'libreuse_streaming_vulkan.so'
        self.host.write_bytes(b'synthetic host library')
        self.android.write_bytes(b'synthetic Android library')
        self.args = ['--generated', str(self.generated), '--host-library', str(self.host),
                     '--android-library', str(self.android)]
        self.sdk = self.repo/'android/reuse-streaming/src/main/kotlin/io/riftco/reusestreaming/ExperimentalVulkanStreaming.kt'
        self.sdk.parent.mkdir(parents=True)
        self.sdk.write_text(''.join(f'internal const val {name} = "unchanged"\n' for name in [
            'BUNDLE_MANIFEST_SHA256', 'ANDROID_LIBRARY_SHA256', 'HOST_LIBRARY_SHA256']))
        self.sdk_before = self.sdk.read_bytes()
        self.build = self.repo/'android/benchmarks/streaming/app/build'
        self.active = self.build/'benchmark-vulkan-assets/vulkan'
        self.active.mkdir(parents=True)
        (self.active/'existing-sentinel').write_bytes(b'unchanged active assets')
        self.build_before = self.snapshot_build()
        self.make_synthetic_package()
        self.repo_patch = patch.object(target, 'REPO', self.repo)
        self.repo_patch.start()
        self.addCleanup(self.repo_patch.stop)

    def snapshot_build(self):
        return {str(p.relative_to(self.build)): p.read_bytes()
                for p in self.build.rglob('*') if p.is_file()}

    def make_synthetic_package(self):
        # Only exercise packaging structure/policy; these bytes cannot run a model.
        plan = b'RVKPLAN1' + struct.pack('<6I', 1, 1, 530, 33, 33, 31)
        plan += bytes(66 * 8 + 31 * 12)
        plan += struct.pack('<33I', 1, *([0] * 32)) * 530
        (self.generated/'plan.bin').write_bytes(plan)
        (self.generated/'weights.bin').write_bytes(bytes(4))
        export = {
            'layers': 8, 'stateCount': 31, 'precision': 'float32',
            'checkpointSha256': target.CHECKPOINT, 'baselineModelSha256': target.BASELINE,
            'sourceModelSha256': target.SOURCE, 'selectedWeightBytesUnchanged': True,
            'recordWords': 33, 'layoutFusion': True, 'dispatchCount': 530, 'arenaBytes': 4,
            'files': {name: target.pin(self.generated/name) for name in ['plan.bin', 'weights.bin']},
        }
        (self.generated/'export.json').write_text(json.dumps(export))
        shaders = self.generated/'shaders'
        shaders.mkdir()
        kernels = {}
        for i in range(12):
            name = f'synthetic_{i}'
            spv = shaders/(name+'.spv')
            spv.write_bytes(f'synthetic shader {i}'.encode())
            kernels[name] = {'floatTypes': [32], 'spirvSha256': target.digest(spv)}
        (shaders/'shader-receipt.json').write_text(json.dumps({
            'float16': False, 'relaxedPrecision': False, 'kernels': kernels,
        }))

    def run_package(self, *flags):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return target.main(self.args + list(flags))

    def assert_no_activation(self):
        self.assertEqual(self.sdk.read_bytes(), self.sdk_before)
        self.assertEqual(self.snapshot_build(), self.build_before)

    @staticmethod
    def approved_receipt():
        return {
            'bundleManifest': {'sha256': target.SUPPORTED_MANIFEST},
            'androidLibrary': {'sha256': target.SUPPORTED_ANDROID_LIBRARY},
            'hostLibrary': {'sha256': target.SUPPORTED_HOST_LIBRARY},
        }

    def test_measured_v2_and_both_exact_libraries_accepted(self):
        target.require_supported_package(self.approved_receipt())

    def test_old_v1_and_new_v3_manifests_rejected(self):
        for rejected in [
            'a2626151ecc5d1f1a5021e6a12d5bb0d691b03f70e5c3fe5a21c9c33f3a1a360',
            '744eb7fbf38da57ea04f37ce7de821037c28bc8a759820eb3098d2803a8c0d0d',
        ]:
            with self.subTest(manifest=rejected):
                receipt = self.approved_receipt()
                receipt['bundleManifest']['sha256'] = rejected
                with self.assertRaisesRegex(ValueError, 'bundleManifest'):
                    target.require_supported_package(receipt)

    def test_missing_or_changed_native_artifact_never_approved(self):
        for key in self.approved_receipt():
            for replacement in [None, {'sha256': '0' * 64}, {}]:
                with self.subTest(key=key, replacement=replacement):
                    receipt = copy.deepcopy(self.approved_receipt())
                    if replacement is None:
                        del receipt[key]
                    else:
                        receipt[key] = replacement
                    with self.assertRaises(ValueError):
                        target.require_supported_package(receipt)

    def test_unapproved_package_cannot_stage(self):
        with self.assertRaisesRegex(ValueError, 'not the measured v2'):
            self.run_package()
        self.assert_no_activation()

    def test_pin_update_flag_cannot_approve_new_package(self):
        with self.assertRaisesRegex(ValueError, 'not the measured v2'):
            self.run_package('--update-sdk-pins')
        self.assert_no_activation()

    def test_research_package_only_writes_local_receipts(self):
        self.run_package('--package-only')
        receipt = json.loads((self.generated/'package-receipt.json').read_text())
        self.assertEqual(receipt['bundleManifest'], target.pin(self.generated/'manifest.json'))
        self.assertEqual(receipt['androidLibrary'], target.pin(self.android))
        self.assertEqual(receipt['hostLibrary'], target.pin(self.host))
        self.assert_no_activation()

    def test_package_only_and_sdk_pin_update_are_incompatible(self):
        with self.assertRaises(SystemExit) as error:
            self.run_package('--package-only', '--update-sdk-pins')
        self.assertEqual(error.exception.code, 2)
        self.assertFalse((self.generated/'manifest.json').exists())
        self.assert_no_activation()

    def test_research_cannot_write_manifest_in_active_directory(self):
        self.args[1] = str(self.active)
        with self.assertRaises(SystemExit) as error:
            self.run_package('--package-only')
        self.assertEqual(error.exception.code, 2)
        self.assert_no_activation()

    def test_exact_pins_reach_staging_and_idempotent_sdk_pin_update(self):
        # Simulate exact approved artifacts at the hash boundary only. Synthetic
        # files never establish model correctness or a new trusted hash.
        actual_digest = target.digest
        approved = {self.generated/'manifest.json': target.SUPPORTED_MANIFEST,
                    self.android: target.SUPPORTED_ANDROID_LIBRARY,
                    self.host: target.SUPPORTED_HOST_LIBRARY}
        def fixture_digest(path):
            return approved.get(path) or actual_digest(path)
        with patch.object(target, 'digest', side_effect=fixture_digest):
            self.run_package('--update-sdk-pins')
        self.assertEqual((self.active/'manifest.json').read_bytes(),
                         (self.generated/'manifest.json').read_bytes())
        self.assertEqual((self.active/'plan.bin').read_bytes(),
                         (self.generated/'plan.bin').read_bytes())
        self.assertEqual((self.build/'benchmark-vulkan-libs/arm64-v8a'/self.android.name).read_bytes(),
                         self.android.read_bytes())
        source = self.sdk.read_text()
        for expected in approved.values():
            self.assertIn(expected, source)
        self.assertEqual(hashlib.sha256((self.build/'benchmark-vulkan-assets/validation/vulkan-benchmark.f32').read_bytes()).hexdigest(),
                         'e3150c5d89ff157b323627de4e5ede75a35be2e44088b284daa9d47268ccb9ca')


if __name__ == '__main__':
    unittest.main()
