"""Asset trust-chain tests using synthetic catalogs and mocked HTTP only."""

from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import urllib.error
import warnings
import zipfile

import fetch_streaming_assets as fetcher


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


def record(name, data):
    return {"file": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def zip_bytes(entries):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return output.getvalue()


def synthetic_catalog():
    """Mirror the public receipt structure without weights or generated fixtures."""
    catalog = {"schemaVersion": 1, "releaseTag": fetcher.RELEASE_TAG,
               "supportedLayers": list(range(3, 13)), "precision": "float32",
               "license": "NSCLv1", "researchOnly": True,
               "checkpointSha256": "a" * 64, "models": [], "parityMetadataFiles": {}}
    assets, entries = {}, {}
    for depth in range(3, 13):
        model_name = f"reuse-streaming-l{depth}-fp32.onnx"
        model = f"Synthetic transport fixture, not an ONNX model: depth {depth}".encode()
        model_record = record(model_name, model)
        manifest_name = f"reuse-streaming-l{depth}-fp32.manifest.json"
        manifest = encoded({"layers": depth, "modelSha256": model_record["sha256"],
                            "checkpointSha256": catalog["checkpointSha256"]})
        manifest_record = record(manifest_name, manifest)
        assets.update({model_name: model, manifest_name: manifest})
        raw = {name: bytes([depth]) * size for name, size in {
            "spectral-magnitude.f32": 3852, "spectral-phase.f32": 3852,
            "spectral-enhanced-magnitude.f32": 3852, "spectral-enhanced-phase.f32": 3852,
            "640-input.f32": 2560, "640-output.f32": 2560}.items()}
        receipt = {"schemaVersion": 1, "passed": True, "layers": depth,
                   "modelSha256": model_record["sha256"], "checkpointSha256": catalog["checkpointSha256"],
                   "dtype": "float32", "spectral": {"frames": 3,
                       "magnitudeFile": "spectral-magnitude.f32", "phaseFile": "spectral-phase.f32",
                       "enhancedMagnitudeFile": "spectral-enhanced-magnitude.f32",
                       "enhancedPhaseFile": "spectral-enhanced-phase.f32"},
                   "fixtures": [{"samples": 640, "inputFile": "640-input.f32", "outputFile": "640-output.f32"}],
                   "files": {name: record(name, data) for name, data in raw.items()}}
        receipt_name = f"l{depth}/parity.json"
        receipt_data = encoded(receipt)
        receipt_record = record(receipt_name, receipt_data)
        row = {"layers": depth, "spectralFrames": 3, "audioSampleCounts": [640]}
        for prefix, item in (("model", model_record), ("manifest", manifest_record), ("parity", receipt_record)):
            row.update({prefix + key.title(): value for key, value in item.items()})
        catalog["models"].append(row)
        entries[receipt_name] = receipt_data
        entries.update({f"l{depth}/{name}": data for name, data in raw.items()})
    for name, data in {"LICENSE": b"Synthetic NSCL metadata fixture", "NOTICE": b"Synthetic attribution fixture",
                       "MAMBA-LICENSE.txt": b"Synthetic Mamba notice fixture", "source.json": b'{"synthetic":true}'}.items():
        entries[name] = data
        catalog["parityMetadataFiles"][name] = record(name, data)
    archive_name = "reuse-streaming-parity.zip"
    assets[archive_name] = zip_bytes(entries)
    catalog["parityArchive"] = record(archive_name, assets[archive_name])
    return catalog, assets, entries


class StreamingAssetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.catalog, self.assets, self.entries = synthetic_catalog()
        self.archive = self.root / "parity.zip"
        self.destination = self.root / "extracted"
        self.downloads = self.root / "downloads"
        self.downloads.mkdir()

    def extract(self):
        self.archive.write_bytes(zip_bytes(self.entries))
        return fetcher.extract_parity(self.archive, self.destination, self.catalog)

    def rejected_archive(self, message):
        with self.assertRaisesRegex(ValueError, message):
            self.extract()
        self.assertFalse(self.destination.exists(), "Failed archive validation must not write fixtures")

    def update_receipt(self, change, update_pin=True, depth=3):
        name = f"l{depth}/parity.json"
        receipt = json.loads(self.entries[name])
        change(receipt)
        self.entries[name] = encoded(receipt)
        if update_pin:
            row = self.catalog["models"][depth - 3]
            item = record(name, self.entries[name])
            row.update(parityBytes=item["bytes"], paritySha256=item["sha256"])

    def fake_http(self, request, timeout):
        self.assertTrue(request.full_url.startswith(fetcher.RELEASE_URL))
        self.assertIsNone(request.get_header("Authorization"))
        self.assertEqual(timeout, 120)
        return io.BytesIO(self.assets[request.full_url[len(fetcher.RELEASE_URL):]])

    def test_valid_all_depth_archive_extracts_every_pinned_fixture_and_notice(self):
        self.assertEqual(self.extract(), 74)
        for name, data in self.entries.items():
            self.assertEqual((self.destination / name).read_bytes(), data)

    def test_corrupt_raw_bytes_or_lengths_are_rejected_before_any_write(self):
        for corrupt in (b"x" * 2560, b"short"):
            with self.subTest(bytes=len(corrupt)):
                self.entries["l3/640-input.f32"] = corrupt
                self.rejected_archive("size/SHA-256")

    def test_extra_or_missing_raw_entries_are_rejected(self):
        original = dict(self.entries)
        for name, extra in (("l3/640-input.f32", False), ("l3/unknown.f32", True)):
            with self.subTest(name=name):
                self.entries = dict(original)
                if extra:
                    self.entries[name] = b"extra"
                else:
                    self.entries.pop(name)
                self.rejected_archive("Unexpected or missing parity archive files")

    def test_missing_receipt_or_metadata_is_rejected(self):
        original = dict(self.entries)
        for name, message in (("l4/parity.json", "Missing parity receipt"),
                              ("LICENSE", "Unexpected or missing parity archive files")):
            with self.subTest(name=name):
                self.entries = dict(original)
                self.entries.pop(name)
                self.rejected_archive(message)

    def test_duplicate_paths_are_rejected(self):
        with zipfile.ZipFile(self.archive, "w") as archive, warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            archive.writestr("l3/parity.json", b"first")
            archive.writestr("l3/parity.json", b"second")
        with self.assertRaisesRegex(ValueError, "Duplicate parity archive paths"):
            fetcher.extract_parity(self.archive, self.destination, self.catalog)
        self.assertFalse(self.destination.exists())

    def test_invalid_zip_and_crc_corruption_fail_before_fixture_writes(self):
        self.archive.write_bytes(b"invalid ZIP fixture")
        with self.assertRaises(zipfile.BadZipFile):
            fetcher.extract_parity(self.archive, self.destination, self.catalog)
        with zipfile.ZipFile(self.archive, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, data in self.entries.items():
                archive.writestr(name, data)
        with zipfile.ZipFile(self.archive) as archive:
            entry = archive.getinfo("l3/640-input.f32")
            offset = entry.header_offset + 30 + len(entry.filename.encode()) + len(entry.extra)
        corrupted = bytearray(self.archive.read_bytes())
        corrupted[offset] ^= 1
        self.archive.write_bytes(corrupted)
        with self.assertRaises(zipfile.BadZipFile):
            fetcher.extract_parity(self.archive, self.destination, self.catalog)
        self.assertFalse(self.destination.exists())

    def test_absolute_traversal_windows_directory_and_symlink_entries_are_rejected(self):
        for filename, mode in (("/outside.f32", None), ("l3/../outside.f32", None),
                               ("l3\\outside.f32", None), ("l3/", None), ("l3/link.f32", 0o120777)):
            with self.subTest(filename=filename):
                with zipfile.ZipFile(self.archive, "w") as archive:
                    info = zipfile.ZipInfo(filename)
                    if mode:
                        info.external_attr = mode << 16
                    archive.writestr(info, b"unsafe")
                with self.assertRaisesRegex(ValueError, "Unsafe parity archive entry"):
                    fetcher.extract_parity(self.archive, self.destination, self.catalog)
                self.assertFalse(self.destination.exists())

    def test_oversized_member_and_expansion_or_entry_count_limits_are_rejected(self):
        self.entries["large.f32"] = b"x" * (2 * 1024 * 1024 + 1)
        self.rejected_archive("Unsafe parity archive entry")
        for count, size in ((257, 1), (256, 300000)):
            archive = MagicMock()
            archive.__enter__.return_value = archive
            archive.infolist.return_value = [SimpleNamespace(filename=f"{i}.f32", file_size=size) for i in range(count)]
            with patch.object(fetcher.zipfile, "ZipFile", return_value=archive), self.assertRaisesRegex(ValueError, "expansion limit"):
                fetcher.extract_parity(self.archive, self.destination, self.catalog)
            archive.read.assert_not_called()
            self.assertFalse(self.destination.exists())

    def test_changed_receipt_cannot_become_its_own_root_of_trust(self):
        self.update_receipt(lambda receipt: receipt.update(modelSha256="b" * 64), update_pin=False)
        self.rejected_archive("size/SHA-256")

    def test_pinned_receipt_identity_must_match_the_catalog(self):
        original = dict(self.entries)
        catalog = copy.deepcopy(self.catalog)
        for field, value in (("schemaVersion", 2), ("passed", False), ("layers", 4),
                             ("modelSha256", "b" * 64), ("checkpointSha256", "c" * 64), ("dtype", "float16")):
            with self.subTest(field=field):
                self.entries, self.catalog = dict(original), copy.deepcopy(catalog)
                self.update_receipt(lambda receipt: receipt.__setitem__(field, value))
                self.rejected_archive("Parity receipt identity mismatch")

    def test_receipt_case_coverage_and_nonempty_file_set_are_required(self):
        original = dict(self.entries)
        catalog = copy.deepcopy(self.catalog)
        changes = [(lambda receipt: receipt["spectral"].update(frames=2), "Incomplete streaming reference cases"),
                   (lambda receipt: receipt.update(fixtures=[]), "Incomplete streaming reference cases"),
                   (lambda receipt: receipt.update(files={}), "Empty parity fixture set")]
        for change, message in changes:
            self.entries, self.catalog = dict(original), copy.deepcopy(catalog)
            self.update_receipt(change)
            self.rejected_archive(message)

    def test_receipt_fixture_path_and_catalog_receipt_path_are_checked(self):
        self.update_receipt(lambda receipt: receipt["files"].update({"../escape.f32": record("../escape.f32", b"bad")}))
        self.rejected_archive("Invalid parity fixture path")
        self.catalog["models"][0]["parityFile"] = "other/parity.json"
        self.rejected_archive("Unexpected parity receipt path")

    def test_metadata_bytes_and_trust_records_cannot_be_substituted(self):
        self.entries["LICENSE"] = b"changed license"
        self.rejected_archive("size/SHA-256")
        self.catalog["parityMetadataFiles"]["l3/parity.json"] = record("l3/parity.json", self.entries["l3/parity.json"])
        self.rejected_archive("Duplicate parity metadata path")

    def test_download_checks_exact_size_hash_and_bounded_read_before_writing(self):
        data = b"pinned asset"
        item = record("model.onnx", data)
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = data
        with patch.object(fetcher.urllib.request, "urlopen", return_value=response):
            path = fetcher.download(item, self.downloads)
        response.read.assert_called_once_with(len(data) + 1)
        self.assertEqual(path.read_bytes(), data)
        path.unlink()
        for bad in (data + b"!", data[:-1], b"x" * len(data)):
            with self.subTest(bytes=len(bad)), patch.object(fetcher.urllib.request, "urlopen", return_value=io.BytesIO(bad)):
                with self.assertRaisesRegex(ValueError, "size/SHA-256"):
                    fetcher.download(item, self.downloads)
            self.assertFalse(path.exists())

    def test_download_rejects_unsafe_name_size_and_network_failure(self):
        item = record("model.onnx", b"asset")
        for filename in ("../model.onnx", "/model.onnx", "dir/model.onnx", "dir\\model.onnx"):
            with self.subTest(filename=filename), patch.object(fetcher.urllib.request, "urlopen") as http:
                with self.assertRaisesRegex(ValueError, "Invalid release asset name"):
                    fetcher.download(dict(item, file=filename), self.downloads)
                http.assert_not_called()
        for size in (0, 32 * 1024 * 1024 + 1):
            with self.subTest(size=size), patch.object(fetcher.urllib.request, "urlopen") as http:
                with self.assertRaisesRegex(ValueError, "size limit"):
                    fetcher.download(dict(item, bytes=size), self.downloads)
                http.assert_not_called()
        with patch.object(fetcher.urllib.request, "urlopen", side_effect=urllib.error.URLError("synthetic offline failure")):
            with self.assertRaises(urllib.error.URLError):
                fetcher.download(item, self.downloads)
        self.assertFalse((self.downloads / item["file"]).exists())

    def test_invalid_trust_record_hash_and_size_are_rejected(self):
        item = record("fixture.f32", b"1234")
        for malformed in (dict(item, bytes=0), dict(item, sha256="not-a-sha256"), dict(item, sha256="A" * 64)):
            with self.subTest(record=malformed), self.assertRaises(ValueError):
                fetcher.checked_bytes(b"1234", malformed, "synthetic fixture")

    def test_fetch_checks_all_twenty_one_downloads_without_any_credentials(self):
        output = self.root / "assets"
        with patch.object(fetcher.urllib.request, "urlopen", side_effect=self.fake_http) as http:
            self.assertEqual(fetcher.fetch(output, self.catalog), 74)
        self.assertEqual(http.call_count, 21)
        self.assertEqual(len(list((output / "models").iterdir())), 20)
        for depth in range(3, 13):
            self.assertTrue((output / "parity" / f"l{depth}" / "parity.json").is_file())

    def test_invalid_catalog_and_existing_output_fail_before_downloads(self):
        original = copy.deepcopy(self.catalog)
        for field, value in (("schemaVersion", 2), ("releaseTag", "untrusted-tag"),
                             ("supportedLayers", [4, 8]), ("precision", "float16"),
                             ("license", "OpenMDW-1.1"), ("researchOnly", False), ("models", self.catalog["models"][:1])):
            with self.subTest(field=field), patch.object(fetcher.urllib.request, "urlopen") as http:
                catalog = copy.deepcopy(original)
                catalog[field] = value
                with self.assertRaisesRegex(ValueError, "Invalid trusted streaming catalog"):
                    fetcher.fetch(self.root / "unused", catalog)
                http.assert_not_called()
        with patch.object(fetcher.urllib.request, "urlopen") as http, self.assertRaisesRegex(ValueError, "fresh fixture output"):
            fetcher.fetch(self.downloads, original)
        http.assert_not_called()

    def test_pinned_manifest_must_match_model_layer_and_checkpoint_identity(self):
        original_catalog, original_assets = copy.deepcopy(self.catalog), dict(self.assets)
        for field, value in (("modelSha256", "b" * 64), ("layers", 4), ("checkpointSha256", "c" * 64)):
            with self.subTest(field=field):
                self.catalog, self.assets = copy.deepcopy(original_catalog), dict(original_assets)
                row = self.catalog["models"][0]
                name = row["manifestFile"]
                manifest = json.loads(self.assets[name])
                manifest[field] = value
                self.assets[name] = encoded(manifest)
                row.update(manifestBytes=len(self.assets[name]), manifestSha256=hashlib.sha256(self.assets[name]).hexdigest())
                with patch.object(fetcher.urllib.request, "urlopen", side_effect=self.fake_http):
                    with self.assertRaisesRegex(ValueError, "Model manifest identity mismatch"):
                        fetcher.fetch(self.root / ("wrong-" + field), self.catalog)

    def test_downloaded_manifest_cannot_replace_the_checked_in_manifest_pin(self):
        row = self.catalog["models"][0]
        self.assets[row["manifestFile"]] = encoded({"layers": 3, "modelSha256": "b" * 64,
                                                  "checkpointSha256": self.catalog["checkpointSha256"]})
        with patch.object(fetcher.urllib.request, "urlopen", side_effect=self.fake_http):
            with self.assertRaisesRegex(ValueError, "size/SHA-256"):
                fetcher.fetch(self.root / "bad-manifest", self.catalog)

    def test_cli_writes_only_validated_assets_path_to_github_environment(self):
        catalog_file = self.root / "catalog.json"
        catalog_file.write_bytes(encoded(self.catalog))
        output, environment = self.root / "cli-assets", self.root / "github-env"
        with patch.object(fetcher, "CATALOG", catalog_file), patch.object(sys, "argv", [
                "fetch_streaming_assets.py", "--output", str(output), "--github-env", str(environment)]), \
                patch.object(fetcher.urllib.request, "urlopen", side_effect=self.fake_http), redirect_stdout(io.StringIO()):
            fetcher.main()
        self.assertEqual(environment.read_text(), f"REUSE_STREAMING_ASSETS={output.resolve()}\n")


if __name__ == "__main__":
    unittest.main()
