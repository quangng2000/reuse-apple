"""Local artifact-policy tests; fixture signatures never invoke real GPG."""

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import central_bundle as bundler


VERSION = "0.1.2"
ARMOR = b"-----BEGIN PGP SIGNATURE-----\nfixture\n-----END PGP SIGNATURE-----\n"


def write_zip(path, entries):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)


def change_zip(path, change):
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    change(entries)
    write_zip(path, entries)


def publication(repository, artifact_id, version=VERSION):
    """Create complete, synthetic structural artifacts using real public notices."""
    profile = bundler.PROFILES[artifact_id]
    release = repository / Path(*bundler.GROUP.split("."), artifact_id, version)
    release.mkdir(parents=True)
    base = artifact_id + "-" + version
    paths = {suffix: release / (base + suffix)
             for suffix in (".aar", ".pom", ".module", "-sources.jar", "-javadoc.jar")}
    licenses = {"licenses/" + source.name: source.read_bytes()
                for source in profile["assets"].iterdir() if source.is_file()}
    classes = io.BytesIO()
    with zipfile.ZipFile(classes, "w") as archive:
        archive.writestr("io/riftco/" + ("reusestreaming/StreamingEnhancer.class" if artifact_id == bundler.STREAMING_ARTIFACT
                                       else "reuseaudio/ReuseEnhancer.class"), b"\xca\xfe\xba\xbefixture")
    aar = {profile["asset_prefix"] + Path(name).name: data for name, data in licenses.items()}
    aar["classes.jar"] = classes.getvalue()
    aar.update({f"jni/{abi}/{profile['library']}": b"\x7fELFsynthetic native library" for abi in bundler.ABIS})
    write_zip(paths[".aar"], aar)
    sources = dict(licenses)
    for name in profile["api"]:
        sources[name] = profile["package"] + b"\n/** Public API KDoc fixture. */\n" + b"// API declaration fixture\n" * 8
    for name in (profile["adapter"], "pocketfft_hdronly.h"):
        sources["native/" + name] = b"// Native source fixture\n" * 8
    write_zip(paths["-sources.jar"], sources)
    documentation = dict(licenses)
    documentation["README.md"] = b"Android SDK integration and research licensing instructions. " * 5
    documentation.update({"api/" + name: sources[name] for name in profile["api"]})
    write_zip(paths["-javadoc.jar"], documentation)
    license_name = "NVIDIA One-Way Noncommercial License v1 (NSCLv1)" if artifact_id == bundler.STREAMING_ARTIFACT else "OpenMDW-1.1"
    paths[".pom"].write_text(
        '<project xmlns="http://maven.apache.org/POM/4.0.0"><modelVersion>4.0.0</modelVersion>'
        f'<groupId>{bundler.GROUP}</groupId><artifactId>{artifact_id}</artifactId><version>{version}</version>'
        '<packaging>aar</packaging><name>Android SDK</name><description>Public API</description>'
        '<url>https://github.com/quangng2000/reuse-audio</url>'
        f'<licenses><license><name>{license_name}</name><url>https://example.invalid/LICENSE</url></license></licenses>'
        '<developers><developer><name>quangng2000</name><url>https://github.com/quangng2000</url></developer></developers>'
        '<scm><connection>scm:git:https://github.com/quangng2000/reuse-audio.git</connection>'
        '<developerConnection>scm:git:ssh://git@github.com/quangng2000/reuse-audio.git</developerConnection>'
        '<url>https://github.com/quangng2000/reuse-audio</url></scm>'
        '<dependencies><dependency><groupId>com.microsoft.onnxruntime</groupId><artifactId>onnxruntime-android</artifactId>'
        '<version>1.23.2</version><scope>runtime</scope></dependency></dependencies></project>')
    files = [{"url": path.name, "size": path.stat().st_size,
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
             for suffix, path in paths.items() if suffix in (".aar", "-sources.jar", "-javadoc.jar")]
    paths[".module"].write_text(json.dumps({"component": {"group": bundler.GROUP, "module": artifact_id,
                                                        "version": version}, "variants": [{"files": files}]}))
    return paths


class BundlePolicyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repository = self.root / "repository"
        self.offline = publication(self.repository, bundler.ARTIFACT)
        self.streaming = publication(self.repository, bundler.STREAMING_ARTIFACT)
        self.args = SimpleNamespace(repository=self.repository, output=self.root / "bundle.zip", version=VERSION,
                                    include_streaming=True, signing_key=None, gpg_home=None,
                                    passphrase_file=None, allow_unsigned=True)

    def validate(self):
        return bundler.bundle(self.args)

    def mark_signed(self):
        for paths in (self.offline, self.streaming):
            for path in paths.values():
                Path(str(path) + ".asc").write_bytes(ARMOR)

    def test_unsigned_cohort_validates_both_licensed_profiles_without_a_zip(self):
        result = self.validate()
        self.assertEqual(result["artifactCount"], 10)
        self.assertEqual(result["coordinateSet"], [f"{bundler.GROUP}:{artifact}:{VERSION}" for artifact in bundler.PROFILES])
        self.assertFalse(result["uploadReady"])
        self.assertFalse(result["bundleCreated"])
        self.assertFalse(self.args.output.exists())

    def test_singleton_default_preserves_legacy_coordinates_and_optional_module(self):
        self.args.include_streaming = False
        self.offline[".module"].unlink()
        result = self.validate()
        self.assertEqual(result["artifactCount"], 4)
        self.assertEqual(result["coordinates"], f"{bundler.GROUP}:{bundler.ARTIFACT}:{VERSION}")

    def test_all_cohort_components_are_validated_before_any_gpg_operation(self):
        self.streaming[".aar"].unlink()
        self.args.signing_key = "synthetic-key-identifier"
        with patch.object(bundler, "gpg_command") as gpg, self.assertRaisesRegex(ValueError, "artifact is missing"):
            self.validate()
        gpg.assert_not_called()

    def test_mixed_versions_and_missing_module_are_rejected(self):
        self.streaming[".module"].unlink()
        with self.assertRaisesRegex(ValueError, "module metadata"):
            self.validate()
        self.streaming[".pom"].write_text(self.streaming[".pom"].read_text().replace(f"<version>{VERSION}</version>", "<version>0.1.3</version>"))
        # Restore a module so the coordinate check itself is reached.
        self.streaming[".module"].write_text('{}')
        with self.assertRaisesRegex(ValueError, "POM version"):
            self.validate()

    def test_streaming_pom_cannot_claim_the_offline_license(self):
        path = self.streaming[".pom"]
        path.write_text(path.read_text().replace("NVIDIA One-Way Noncommercial License v1 (NSCLv1)", "OpenMDW-1.1"))
        with self.assertRaisesRegex(ValueError, "streaming NSCLv1"):
            self.validate()

    def test_offline_pom_cannot_apply_streaming_license(self):
        path = self.offline[".pom"]
        path.write_text(path.read_text().replace("OpenMDW-1.1", "NVIDIA One-Way Noncommercial License v1 (NSCLv1)"))
        with self.assertRaisesRegex(ValueError, "offline OpenMDW"):
            self.validate()

    def test_artifacts_cannot_transitively_link_the_other_license_boundary(self):
        path = self.offline[".pom"]
        dependency = f'<dependency><groupId>{bundler.GROUP}</groupId><artifactId>{bundler.STREAMING_ARTIFACT}</artifactId><version>{VERSION}</version></dependency>'
        path.write_text(path.read_text().replace('</dependencies>', dependency + '</dependencies>'))
        with self.assertRaisesRegex(ValueError, "remain independent"):
            self.validate()

    def test_streaming_classes_and_license_cannot_enter_the_offline_aar(self):
        change_zip(self.offline[".aar"], lambda entries: entries.update({"assets/reuse-audio/licenses/NSCL.txt": bundler.STREAMING_LICENSE.read_bytes()}))
        with self.assertRaisesRegex(ValueError, "Streaming material"):
            self.validate()

    def test_hidden_nscl_license_in_offline_aar_is_rejected(self):
        change_zip(self.offline[".aar"], lambda entries: entries.update({"assets/reuse-audio/licenses/extra.txt": bundler.STREAMING_LICENSE.read_bytes()}))
        with self.assertRaisesRegex(ValueError, "must not contain the streaming license"):
            self.validate()

    def test_license_bytes_and_native_abi_coverage_are_strict(self):
        change_zip(self.streaming[".aar"], lambda entries: entries.__setitem__("assets/reuse-streaming/licenses/LICENSE.txt", b"incorrect license"))
        with self.assertRaisesRegex(ValueError, "differs from the checked-in"):
            self.validate()
        self.streaming = publication(self.root / "other", bundler.STREAMING_ARTIFACT)
        for paths, artifact_id in ((self.offline, bundler.ARTIFACT), (self.streaming, bundler.STREAMING_ARTIFACT)):
            with self.subTest(artifact=artifact_id):
                library = bundler.PROFILES[artifact_id]["library"]
                change_zip(paths[".aar"], lambda entries: entries.pop("jni/x86/" + library))
                with self.assertRaisesRegex(ValueError, "four supported ABIs"):
                    bundler.verify_archives(paths[".aar"], paths["-sources.jar"], paths["-javadoc.jar"], artifact_id)

    def test_models_and_private_files_are_rejected_in_both_profiles(self):
        for artifact_id in bundler.PROFILES:
            for name, data in (("weights.onnx", b"model"), ("credentials.json", b"token"),
                               ("note.txt", b"-----BEGIN PGP PRIVATE KEY BLOCK-----")):
                with self.subTest(artifact=artifact_id, filename=name), self.assertRaises(ValueError):
                    bundler.safe_content(name, data, artifact_id)

    def test_missing_api_native_sources_or_documentation_rejects_publication(self):
        change_zip(self.streaming["-sources.jar"], lambda entries: entries.pop("native/pocketfft_hdronly.h"))
        with self.assertRaisesRegex(ValueError, "native DSP sources"):
            bundler.verify_archives(self.streaming[".aar"], self.streaming["-sources.jar"], self.streaming["-javadoc.jar"], bundler.STREAMING_ARTIFACT)
        change_zip(self.offline["-javadoc.jar"], lambda entries: entries.pop("README.md"))
        with self.assertRaisesRegex(ValueError, "README.md"):
            self.validate()

    def test_module_cannot_reference_wrong_coordinates_or_modified_artifacts(self):
        module = json.loads(self.streaming[".module"].read_text())
        module["component"]["version"] = "0.1.3"
        self.streaming[".module"].write_text(json.dumps(module))
        with self.assertRaisesRegex(ValueError, "module coordinates"):
            self.validate()
        module["component"]["version"] = VERSION
        module["variants"][0]["files"][0]["sha256"] = "0" * 64
        self.streaming[".module"].write_text(json.dumps(module))
        with self.assertRaisesRegex(ValueError, "module checksum"):
            self.validate()

    def test_unsigned_or_partially_signed_cohort_cannot_make_upload_zip(self):
        self.args.allow_unsigned = False
        with self.assertRaisesRegex(ValueError, "Missing signatures"):
            self.validate()
        self.mark_signed()
        Path(str(self.streaming[".pom"]) + ".asc").unlink()
        self.args.allow_unsigned = True
        with patch.object(bundler, "gpg_command", return_value=["synthetic-gpg"]), patch.object(bundler, "verify_signature"):
            result = self.validate()
        self.assertEqual(result["verifiedSignatureCount"], 9)
        self.assertFalse(result["bundleCreated"])
        self.assertFalse(self.args.output.exists())

    def test_verified_cohort_bundles_deterministically_with_exact_checksum_coverage(self):
        self.mark_signed()
        for paths in (self.offline, self.streaming):
            Path(str(paths[".pom"]) + ".asc.sha256").write_text("must be omitted")
            (paths[".pom"].parent / "maven-metadata.xml").write_text("must be omitted")
        with patch.object(bundler, "gpg_command", return_value=["synthetic-gpg"]), patch.object(bundler, "verify_signature") as verification:
            result = self.validate()
            self.assertEqual(verification.call_count, 10)
            original = self.args.output.read_bytes()
            self.validate()
            self.assertEqual(original, self.args.output.read_bytes())
        self.assertTrue(result["uploadReady"])
        self.assertEqual(result["entryCount"], 60)
        with zipfile.ZipFile(self.args.output) as archive:
            self.assertEqual(archive.namelist(), sorted(archive.namelist()))
            self.assertFalse(any("metadata" in name or ".asc." in name for name in archive.namelist()))
            for name in archive.namelist():
                if name.endswith((".aar", ".pom", ".module", ".jar")):
                    for algorithm in bundler.CHECKSUMS:
                        self.assertEqual(archive.read(name + "." + algorithm).strip().decode(),
                                         hashlib.new(algorithm, archive.read(name)).hexdigest())

    def test_invalid_signature_cannot_create_a_bundle(self):
        self.mark_signed()
        with patch.object(bundler, "gpg_command", return_value=["synthetic-gpg"]), patch.object(bundler.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout=b"")):
            with self.assertRaisesRegex(ValueError, "signature verification failed"):
                self.validate()
        self.assertFalse(self.args.output.exists())


if __name__ == "__main__":
    unittest.main()
