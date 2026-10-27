import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from extract_comparison import CLASS, DEFAULT_CASES, DEFAULT_INPUT_SHA, MODEL_HASHES, verify_bundle


RUN = "comparison-fixture"


def fixture_result():
    cases, plan = [], []
    for index, entry in enumerate(DEFAULT_CASES.split("+")):
        backend, threads = entry.split("@")
        identity = {"backend": backend, "threads": int(threads), "layers": 8}
        plan.append(identity.copy())
        cases.append({**identity, "caseId": f"{index + 1}-{backend}-t{threads}", "success": True,
                      "timedFrames": 300, "modelSha256": MODEL_HASHES[backend],
                      "allOutputFinite": True, "inputSamples": 106880, "outputSamples": 106880,
                      "qualityCapture": False, "timedOutputChecksum": -.5,
                      "rawPushMs": [10.0, 20.0, 30.0, 40.0] * 75,
                      "rawRenderMs": [9.0, 19.0, 29.0, 39.0] * 75,
                      "latencyMs": {"mean": 25.0, "min": 10.0, "p50": 20.0, "p95": 40.0, "p99": 40.0, "max": 40.0},
                      "renderLatencyMs": {"mean": 24.0, "min": 9.0, "p50": 19.0, "p95": 39.0, "p99": 39.0, "max": 39.0},
                      "rtf": 1.25, "over20msCount": 150, "over20msPercent": 50.0, "timedAudioSeconds": 6.0,
                      "fixture": {"passed": True, "resetBitwiseRepeatPassed": True, "samples": 16000,
                                  "maximumAbsoluteError": 1e-8, "rmsError": 2e-9, "absoluteTolerance": .0003}})
    return {"schemaVersion": 1, "runId": RUN, "success": True, "status": "complete",
            "variant": "native-v2-comparison", "sdkSource": "local", "precision": "FP32",
            "executionProvider": "CPUExecutionProvider", "onnxRuntimeVersion": "1.23.2",
            "sdkCoordinate": "io.github.quangng2000:reuse-streaming-android:0.1.2",
            "sampleRate": 16000, "hopSamples": 320, "windowSamples": 640,
            "timedFramesPerCase": 300, "warmupHopsPerCase": 32, "completedCases": 5,
            "inputSha256": DEFAULT_INPUT_SHA, "qualityCapture": False,
            "device": {"targetDebuggable": False, "abis": ["arm64-v8a"], "emulatorReported": False,
                       "fingerprint": "samsung/physical-fixture/api35"}, "cases": cases, "plan": plan}


def write_bundle(folder, result):
    def write(kind, raw):
        path = folder / f"{RUN}-{kind}.json"
        path.write_bytes(raw)
        path.with_suffix(".sha256").write_text(hashlib.sha256(raw).hexdigest() + "  " + path.name + "\n")
    raw = json.dumps(result, separators=(",", ":")).encode()
    write("result", raw)
    receipt = {"schemaVersion": 1, "runId": RUN, "success": True, "instrumentationClass": CLASS,
               "requested": {"comparisonCases": DEFAULT_CASES, "timedFrames": 300, "requirePhysical": True,
                             "captureOutput": False, "expectedInputSha256": DEFAULT_INPUT_SHA}, "exports": [],
               "result": {"file": f"/storage/emulated/0/Android/data/test/files/firebase-comparison/{RUN}-result.json",
                          "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}}
    write("receipt", json.dumps(receipt, separators=(",", ":")).encode())
    return raw


class ExtractComparisonTest(unittest.TestCase):
    def test_ordered_repeated_controls_preserved_and_statistics_recomputed(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            raw = write_bundle(folder, fixture_result())
            summary, files = verify_bundle(folder, RUN)
            self.assertEqual(files["results.json"], raw)
            self.assertEqual([case["backend"] for case in summary["cases"]],
                             ["frequency-v1", "stable-v2", "stable-v2", "stable-v2", "frequency-v1"])
            self.assertEqual([case["threads"] for case in summary["cases"]], [4, 4, 8, 6, 4])
            self.assertEqual(len({case["caseId"] for case in summary["cases"]}), 5)
            self.assertEqual(summary["cases"][0]["latencyMs"]["p50"], 20)
            self.assertEqual(summary["cases"][0]["over20msCount"], 150)
            self.assertEqual(summary["cases"][0]["strictlyUnder20msCount"], 75)

    def test_transport_tamper_is_rejected_even_if_json_remains_valid(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            raw = write_bundle(folder, fixture_result())
            (folder / f"{RUN}-result.json").write_bytes(raw + b" ")
            with self.assertRaisesRegex(ValueError, "integrity"):
                verify_bundle(folder, RUN)

    def test_semantic_corruption_rejected_after_hashes_are_resealed(self):
        changes = {
            "wrong mean": lambda d: d["cases"][0]["latencyMs"].update(mean=26.0),
            "wrong order": lambda d: d["cases"].__setitem__(slice(1, 3), list(reversed(d["cases"][1:3]))),
            "duplicate case": lambda d: d["cases"][1].update(caseId=d["cases"][0]["caseId"]),
            "wrong model": lambda d: d["cases"][1].update(modelSha256=MODEL_HASHES["inline-v2"]),
            "invalid render": lambda d: d["cases"][0]["rawRenderMs"].__setitem__(0, float("nan")),
        }
        for label, change in changes.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as name:
                result = copy.deepcopy(fixture_result())
                change(result)
                folder = Path(name)
                write_bundle(folder, result)
                with self.assertRaises(ValueError):
                    verify_bundle(folder, RUN)

    def test_emulator_is_only_accepted_with_explicit_local_smoke_override(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            result = fixture_result()
            result["device"]["emulatorReported"] = True
            write_bundle(folder, result)
            with self.assertRaisesRegex(ValueError, "Physical target"):
                verify_bundle(folder, RUN)
            summary, _ = verify_bundle(folder, RUN, allow_emulator=True)
            self.assertTrue(summary["device"]["emulatorReported"])


if __name__ == "__main__":
    unittest.main()
