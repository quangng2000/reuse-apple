"""Offline tests: all device/process transport is replaced with in-memory fakes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

import run_device_ablation as runner


IDENTITY = {"candidate_revision": "source-sha256:" + "1" * 64,
            "checkpoint_sha256": runner.benchmark.CHECKPOINT_SHA256, "input_sha256": "2" * 64}


def report(variant, short=True, short_ratio=0.9, continuous_ratio=0.8, failed=False):
    run_id = str(uuid.uuid4()).upper()
    result = {
        "schema_version": 1, "status": "parity_failed" if failed else "completed", "run_id": run_id,
        "baseline_revision": runner.BASELINE, **IDENTITY,
        "layers": 8, "precision": "float32", "lookahead_frames": 0,
        "baseline_options": runner.BASE_OPTIONS, "candidate_options": variant.options,
        "configured_candidate_options": runner.COMBO_OPTIONS,
        "short_only": short, "benchmark_scope": "short_only_ablation" if short else "full_abba",
        "planned_phases": runner.PHASES[:1] if short else runner.PHASES,
        "runtime_override_arguments": variant.arguments(short),
        "optimized_argument_for_both_variants": True, "paced_input": False,
        "require_bitwise_output_equality": True, "output_parity_requirement": "bitwise_equal",
        "absolute_tolerance_is_acceptance_gate": False, "swift_compilation_mode": "wholemodule",
        "sample_rate": 16000, "hop_samples": 320, "output_max_absolute_tolerance": 1e-5,
        "device": {"hardware_model": "test-phone", "os_version": "testOS", "physical_device": True,
                   "low_power_mode": False}, "trials": []}
    for phase in result["planned_phases"]:
        seconds = runner.benchmark.PHASES[phase]
        for index, name in enumerate(runner.benchmark.ORDER):
            did_fail = failed and index == 1
            value = 30 if name == "baseline" else 30 * (short_ratio if short else continuous_ratio)
            if phase == "short-4s" and name == "candidate":
                value = 30 * short_ratio
            row = {"id": f"{phase}-{index}-{name}", "phase": phase, "order_index": index, "variant": name,
                   "run_id": run_id, "engine_options": runner.BASE_OPTIONS if name == "baseline" else variant.options,
                   "audio_seconds": seconds, "output_samples": seconds * 16000, "all_finite": True,
                   "output_sha256": ("c" if did_fail else "b") * 64,
                   "raw_frame_ms": [value] * (seconds * 50 + 1), "frame_timing": {"count": seconds * 50 + 1},
                   "wall_seconds_excluding_load_warmup_io_validation": seconds * 1.5,
                   "thermal_before": "nominal", "thermal_after": "nominal", "thermal_push_counts": {"nominal": 201},
                   "parity": {"passed": not did_fail, "bitwise_equal": not did_fail,
                              "different_float_bit_patterns": 1 if did_fail else 0,
                              "max_abs_difference": 1e-7 if did_fail else 0,
                              "rmse": 1e-8 if did_fail else 0, "relative_rmse": 1e-8 if did_fail else 0,
                              "maximum_absolute_tolerance": 1e-5, "gate": "bitwise_equal",
                              "reference_trial": phase + "-0-baseline"}}
            result["trials"].append(row)
            if did_fail:
                return result
    return result


def feed_events(events, data):
    events.accept({"event": "run_started", "run_id": data["run_id"], "layers": 8,
                   "hardware": data["device"]["hardware_model"], "baseline_revision": runner.BASELINE,
                   "candidate_revision": data["candidate_revision"], "baseline_options": data["baseline_options"],
                   "candidate_options": data["candidate_options"], "short_only": data["short_only"],
                   "planned_phases": data["planned_phases"]})
    for row in data["trials"]:
        events.accept({"event": "trial_started", "run_id": data["run_id"], "trial": row["id"]})
        events.accept({"event": "trial_completed", "run_id": data["run_id"], "trial": row["id"],
                       "parity_passed": row["parity"]["passed"],
                       "different_float_bit_patterns": row["parity"]["different_float_bit_patterns"],
                       "max_abs_difference": row["parity"]["max_abs_difference"]})
    terminal = {"event": "run_completed", "run_id": data["run_id"], "trials": len(data["trials"]),
                "result_file": "Documents/latest-results.json"}
    if data["status"] == "parity_failed":
        terminal = {"event": "failed", "run_id": data["run_id"],
                    "error": "Bitwise output parity failed in " + data["trials"][-1]["id"]}
    events.accept(terminal)


class DeviceAblationTests(unittest.TestCase):
    def test_report_and_live_event_validation_are_strict(self):
        variant = runner.SCREENS[0]
        data = report(variant)
        runner.check_common(data, IDENTITY, variant, True, data["run_id"])
        runner.benchmark.summarize(data)
        events = runner.Events(IDENTITY, variant, True, set())
        feed_events(events, data)
        runner.check_trial_events(data, events)
        for key, wrong in [("precision", "float16"), ("layers", 4), ("baseline_revision", "wrong"),
                           ("candidate_revision", "source-sha256:" + "9" * 64),
                           ("candidate_options", runner.COMBO_OPTIONS), ("run_id", str(uuid.uuid4()))]:
            changed = copy.deepcopy(data)
            changed[key] = wrong
            with self.assertRaises(ValueError, msg=key):
                runner.check_common(changed, IDENTITY, variant, True, data["run_id"])
        with self.assertRaises(ValueError):
            events.accept({"event": "failed", "run_id": str(uuid.uuid4()), "error": "wrong run"})

    def test_only_evidenced_parity_failure_can_be_excluded(self):
        variant = runner.SCREENS[0]
        data = report(variant, failed=True)
        events = runner.Events(IDENTITY, variant, True, set())
        feed_events(events, data)
        runner.validate_parity_failure(data, events, IDENTITY, variant, True)
        changed = copy.deepcopy(data)
        changed["trials"][-1]["parity"]["different_float_bit_patterns"] = 0
        with self.assertRaises(ValueError):
            runner.validate_parity_failure(changed, events, IDENTITY, variant, True)

    def test_threshold_selects_best_short_relative_ratio(self):
        rows = [{"variant": "a", "status": "validated", "timing_eligible": True, "short_ratio": 0.92},
                {"variant": "b", "status": "validated", "timing_eligible": True, "short_ratio": 0.90},
                {"variant": "c", "status": "excluded_parity_failure", "timing_eligible": True, "short_ratio": 0.1}]
        self.assertEqual(runner.best_screen(rows, 0.95)["variant"], "b")
        self.assertIsNone(runner.best_screen(rows, 0.90))

    def run_fake_plan(self, reference=False, parity_failure=False, transport_failure=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            config.write_text("{}")
            config.with_name("candidate-source.json").write_text("{}")
            arguments = ["runner", "--device", "EXPLICIT-DEVICE", "--output", str(root / "results"),
                         "--config", str(config)]
            if reference:
                reference_file = root / "reference.json"
                reference_file.write_text(json.dumps(report(runner.Variant("combo", "decoderFeatures", True),
                                                           short=False, short_ratio=0.95, continuous_ratio=0.8)))
                arguments += ["--reference-report", str(reference_file)]
            captured, current = [], {}
            def fake_launch(command, directory, events, timeout):
                captured.append((events.variant.name, events.short, command))
                if transport_failure:
                    raise ValueError("Device is locked; do not retry")
                data = report(events.variant, events.short,
                              short_ratio={"compiled-only": 0.91, "pool-only": 0.94, "body-plus-pool": 0.89}[events.variant.name],
                              failed=parity_failure and events.variant.name == "compiled-only")
                current["report"] = data
                feed_events(events, data)
            def fake_copy(device, directory, name):
                data = current["report"]
                (directory / name).write_text(json.dumps(data))
                return data
            with patch.object(runner, "load_identity", return_value=IDENTITY), \
                 patch.object(runner, "capture_launch", side_effect=fake_launch), \
                 patch.object(runner, "copy_report", side_effect=fake_copy), patch("sys.argv", arguments):
                code = runner.main()
            return code, captured, json.loads((root / "results/ablation.json").read_text())

    def test_maximum_four_distinct_launches_and_short_reference_rule(self):
        code, calls, manifest = self.run_fake_plan(reference=True)
        self.assertEqual(code, 0)
        self.assertEqual([(name, short) for name, short, _ in calls],
                         [("compiled-only", True), ("pool-only", True), ("body-plus-pool", True), ("body-plus-pool", False)])
        self.assertEqual(manifest["selection"]["variant"], "body-plus-pool")
        for _, _, command in calls:
            self.assertIn("EXPLICIT-DEVICE", command)
            self.assertEqual(command[command.index("--console") + 1], runner.BUNDLE)

    def test_parity_failure_is_excluded_without_retry(self):
        code, calls, manifest = self.run_fake_plan(parity_failure=True)
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 3)
        self.assertEqual(manifest["runs"][0]["status"], "excluded_parity_failure")

    def test_transport_failure_stops_without_another_launch(self):
        code, calls, manifest = self.run_fake_plan(transport_failure=True)
        self.assertEqual(code, 2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(manifest["status"], "stopped")


if __name__ == "__main__":
    unittest.main()
