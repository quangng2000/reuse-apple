#!/usr/bin/env python3
"""Validate and summarize a completed eight-layer FP32 ABBA device benchmark."""

import argparse
import json
import math
from pathlib import Path
import re
import statistics
import sys


BASELINE_REVISION = "bdf56d19f4dbc16fb9ee949d42b70b4b6b5fe43e"
CHECKPOINT_SHA256 = "6eac4e8cc721f99a22cde7eac3c9f8e2826ced5dc8a4234fd1977de1ec0d3ade"
PHASES = {"short-4s": 4, "continuous-32s": 32}
ORDER = ("baseline", "candidate", "candidate", "baseline")
HASH_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def valid_hash(value):
    return isinstance(value, str) and HASH_PATTERN.fullmatch(value) is not None


def latency_stats(values):
    ordered = sorted(values)
    count = len(ordered)
    over = sum(value > 20 for value in ordered)
    return {
        "frames": count,
        "mean_ms": statistics.mean(ordered),
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[math.ceil(0.95 * count) - 1],
        "p99_ms": ordered[math.ceil(0.99 * count) - 1],
        "max_ms": ordered[-1],
        "over_20ms_count": over,
        "over_20ms_fraction": over / count,
        "over_20ms_percent": 100 * over / count,
    }


def validate(report):
    require(isinstance(report, dict), "Results must be a JSON object")
    require(report.get("schema_version") == 1, "Unsupported result schema")
    require(report.get("status") == "completed", "The benchmark has not completed")
    require(report.get("layers") == 8 and report.get("precision") == "float32", "Expected eight layers and FP32")
    require(report.get("baseline_revision") == BASELINE_REVISION, "Baseline revision differs from the pinned reference")
    require(report.get("checkpoint_sha256") == CHECKPOINT_SHA256, "Checkpoint differs from the pinned model")
    candidate = report.get("candidate_revision", "")
    require(isinstance(candidate, str) and candidate.startswith("source-sha256:")
            and valid_hash(candidate.removeprefix("source-sha256:")), "Missing candidate source SHA256")
    require(valid_hash(report.get("input_sha256")), "Missing input SHA256")
    require(report.get("optimized_argument_for_both_variants") is True, "Both variants must use optimized=True")
    require(report.get("sample_rate") == 16000 and report.get("hop_samples") == 320,
            "Expected 16 kHz input and 320-sample hops")
    require(isinstance(report.get("paced_input"), bool), "Missing replay pacing metadata")
    tolerance = report.get("output_max_absolute_tolerance")
    require(finite_number(tolerance) and tolerance > 0, "Invalid output tolerance")
    trials = report.get("trials")
    require(isinstance(trials, list) and len(trials) == 8, "Expected exactly eight completed trials")
    expected_sequence = [(phase, index, variant) for phase in PHASES for index, variant in enumerate(ORDER)]
    observed = [(t.get("phase"), t.get("order_index"), t.get("variant"))
                for t in trials if isinstance(t, dict)]
    require(observed == expected_sequence, "Expected four-second ABBA followed by 32-second ABBA")
    for trial in trials:
        label = trial.get("id", "unnamed trial")
        seconds = PHASES[trial["phase"]]
        require(trial.get("audio_seconds") == seconds, f"{label}: unexpected audio duration")
        require(trial.get("output_samples") == seconds * 16000 and trial.get("all_finite") is True,
                f"{label}: incomplete or nonfinite output")
        require(valid_hash(trial.get("output_sha256")), f"{label}: missing output SHA256")
        raw = trial.get("raw_frame_ms")
        expected_frames = seconds * 50 + 1  # Includes the initial and EOF render frames.
        require(isinstance(raw, list) and len(raw) == expected_frames,
                f"{label}: expected all {expected_frames} raw frame timings")
        require(all(finite_number(value) and value >= 0 for value in raw), f"{label}: invalid raw frame timing")
        reported_count = trial.get("frame_timing", {}).get("count")
        require(reported_count == len(raw), f"{label}: raw/reported frame counts disagree")
        wall = trial.get("wall_seconds_excluding_load_warmup_io_validation")
        require(finite_number(wall) and wall > 0, f"{label}: invalid wall time")
        parity = trial.get("parity")
        require(isinstance(parity, dict) and parity.get("passed") is True, f"{label}: output parity did not pass")
        require(isinstance(parity.get("bitwise_equal"), bool), f"{label}: missing bitwise parity")
        different = parity.get("different_float_bit_patterns")
        require(isinstance(different, int) and not isinstance(different, bool)
                and 0 <= different <= trial["output_samples"], f"{label}: invalid bit difference count")
        require(parity["bitwise_equal"] == (different == 0), f"{label}: inconsistent bitwise parity")
        maximum = parity.get("max_abs_difference")
        require(finite_number(maximum) and 0 <= maximum <= tolerance, f"{label}: invalid output difference")
        for key in ("rmse", "relative_rmse"):
            require(finite_number(parity.get(key)) and parity[key] >= 0, f"{label}: invalid {key}")
        require(parity.get("maximum_absolute_tolerance") == tolerance, f"{label}: changed parity tolerance")
        require(parity.get("reference_trial") == trial["phase"] + "-0-baseline", f"{label}: unexpected parity reference")
        require(isinstance(trial.get("thermal_before"), str) and isinstance(trial.get("thermal_after"), str)
                and isinstance(trial.get("thermal_push_counts"), dict), f"{label}: missing thermal observations")
    for phase in PHASES:
        selected = [trial for trial in trials if trial["phase"] == phase]
        baseline_hash = selected[0]["output_sha256"]
        require(all(not trial["parity"]["bitwise_equal"] or trial["output_sha256"] == baseline_hash
                    for trial in selected), f"{phase}: equal outputs have different hashes")
    return trials


def summarize(report):
    trials = validate(report)
    phases = {}
    trial_summaries = []
    for trial in trials:
        wall = trial["wall_seconds_excluding_load_warmup_io_validation"]
        trial_summaries.append({
            "id": trial["id"], "phase": trial["phase"], "variant": trial["variant"],
            "order_index": trial["order_index"], "audio_seconds": trial["audio_seconds"],
            "latency": latency_stats(trial["raw_frame_ms"]), "wall_seconds": wall,
            "real_time_factor": wall / trial["audio_seconds"],
            "thermal_before": trial["thermal_before"], "thermal_after": trial["thermal_after"],
            "thermal_push_counts": trial["thermal_push_counts"], "parity": trial["parity"],
            "output_sha256": trial["output_sha256"], "output_samples": trial["output_samples"],
        })
    for phase in PHASES:
        variants = {}
        for variant in ("baseline", "candidate"):
            selected = [trial for trial in trials if trial["phase"] == phase and trial["variant"] == variant]
            raw = [value for trial in selected for value in trial["raw_frame_ms"]]
            wall = sum(trial["wall_seconds_excluding_load_warmup_io_validation"] for trial in selected)
            audio = sum(trial["audio_seconds"] for trial in selected)
            variants[variant] = {
                "trial_count": len(selected), "pooled_latency": latency_stats(raw),
                "total_wall_seconds": wall, "total_audio_seconds": audio,
                "real_time_factor": wall / audio,
                "all_outputs_bitwise_equal": all(trial["parity"]["bitwise_equal"] for trial in selected),
                "maximum_output_absolute_difference": max(trial["parity"]["max_abs_difference"] for trial in selected),
            }
        baseline, candidate = variants["baseline"], variants["candidate"]
        baseline_mean = baseline["pooled_latency"]["mean_ms"]
        candidate_mean = candidate["pooled_latency"]["mean_ms"]
        require(baseline_mean > 0 and candidate_mean > 0, f"{phase}: zero mean latency")
        phases[phase] = {
            "variants": variants,
            "mean_frame_speedup": baseline_mean / candidate_mean,
            "mean_frame_latency_savings_percent": 100 * (1 - candidate_mean / baseline_mean),
            "wall_time_speedup": baseline["real_time_factor"] / candidate["real_time_factor"],
            "wall_time_savings_percent": 100 * (1 - candidate["real_time_factor"] / baseline["real_time_factor"]),
        }
    return {
        "schema_version": 1, "status": "validated", "primary_phase": "continuous-32s",
        "baseline_revision": report["baseline_revision"], "candidate_revision": report["candidate_revision"],
        "checkpoint_sha256": report["checkpoint_sha256"], "input_sha256": report["input_sha256"],
        "device": report.get("device"), "layers": 8, "precision": "float32",
        "paced_input": report["paced_input"], "phases": phases, "trials": trial_summaries,
        "method": {
            "pooling": "All raw frames; phases and variants kept separate; no dropped frames or outliers.",
            "p50": "Median, averaging the middle pair for even counts.",
            "p95_p99": "Nearest-rank ceil(P*N).",
            "rtf": "Sum of replay wall time divided by sum of audio duration; includes outer loop overhead.",
            "savings": "100*(1-candidate/baseline); positive means faster candidate.",
            "caveats": [
                "Unpaced file replay is not live-call latency; over20ms is a slow-frame fraction.",
                "32s uses continuous recurrent state across repeated 4s input, not 32s of unique speech.",
                "ABBA balances order, but two trials per variant do not make frames independent repetitions.",
                "Inspect per-trial thermal conditions and first/last baseline drift before attributing changes to code.",
                "For paced replay, wall RTF includes synthetic arrival waits and is not compute throughput.",
            ],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Copied device Documents/latest-results.json")
    arguments = parser.parse_args()
    try:
        with arguments.results.open() as source:
            report = json.load(source)
        result = summarize(report)
        result["source_results"] = str(arguments.results.resolve())
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"Invalid benchmark results: {error}\n")


if __name__ == "__main__":
    main()
