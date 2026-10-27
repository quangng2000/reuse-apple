import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from extract_results import extract_artifacts, extract_logcat


def result_bytes():
    return json.dumps({
        "runId": "test-pair-baseline", "success": True, "status": "complete",
        "precision": "FP32", "executionProvider": "CPUExecutionProvider",
        "variant": "published", "sdkSource": "published",
        "device": {"targetDebuggable": False, "emulatorReported": False, "abis": ["arm64-v8a"]},
        "cases": [{"caseId": "1-l8-t4", "success": True, "layers": 8, "threads": 4,
                   "timedFrames": 32, "rawPushMs": [15.0] * 32,
                   "allOutputFinite": True, "inputSamples": 21120, "outputSamples": 21120,
                   "fixture": {"passed": True, "resetBitwiseRepeatPassed": True},
                   "qualityCapture": False}],
    }, separators=(",", ":")).encode()


class ExtractResultsTest(unittest.TestCase):
    def test_receipt_integrity_and_raw_json_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "download", root / "verified"
            source.mkdir()
            output.mkdir()
            raw = result_bytes()
            name = "test-pair-baseline-result.json"
            (source / name).write_bytes(raw)
            receipt = {"schemaVersion": 1, "runId": "test-pair-baseline", "success": True,
                       "requested": {"requirePhysical": True},
                       "result": {"file": "/storage/emulated/0/Android/data/example/files/firebase-benchmark/" + name,
                                  "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}}
            (source / "test-pair-baseline-receipt.json").write_text(json.dumps(receipt))
            records = extract_artifacts(source, output)
            self.assertEqual(len(records), 1)
            self.assertEqual((Path(records[0]["verifiedDirectory"]) / "results.json").read_bytes(), raw)
            (source / name).write_bytes(raw + b" ")
            with self.assertRaisesRegex(ValueError, "integrity failure"):
                extract_artifacts(source, output)

    def test_logcat_chunks_with_transport_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = result_bytes()
            payload = base64.b64encode(raw).decode()
            chunks = [payload[index:index + 300] for index in range(0, len(payload), 300)]
            lines = ["I REUSE_BENCH: RUN_START test-pair-baseline"]
            lines += [f"I REUSE_BENCH: RESULT_CHUNK {index + 1}/{len(chunks)}:{value}" for index, value in enumerate(chunks)]
            lines += ["I REUSE_BENCH: RESULT_SHA256:" + hashlib.sha256(raw).hexdigest()]
            log = root / "logcat.txt"
            log.write_text("\n".join(lines))
            output = root / "verified"
            output.mkdir()
            records = extract_logcat(log, output)
            self.assertEqual(records[0]["runId"], "test-pair-baseline")
            self.assertFalse(records[0]["pcmExportsVerified"])
            log.write_text("\n".join(lines[:-2] + lines[-1:]))
            with self.assertRaisesRegex(ValueError, "Truncated"):
                extract_logcat(log, output)


if __name__ == "__main__":
    unittest.main()
