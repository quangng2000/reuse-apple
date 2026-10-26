#!/usr/bin/env python3
"""Bounded, UUID-validated iPhone ablation of the already installed benchmark.

Runs three distinct 4s ABBA screens, then at most one full confirmation. A screen
qualifies only when its candidate/baseline mean ratio is at least 2% lower than
the supplied combo reference's SHORT-phase ratio (ratio <= reference * 0.98).
The full 4s+32s confirmation, when run, is authoritative. No builds, installs,
device discovery, retries, or application termination beyond --terminate-existing
for this benchmark bundle are performed.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time
import uuid

import summarize as benchmark

BUNDLE = "io.riftco.reuse-streaming-eight-benchmark"
MARKER = "REUSE_EIGHT_BENCHMARK_EVENT="
BASELINE = "2446c8383ac65da5376d9159fa1d079535c4e36b"
BASE_OPTIONS = {"compilation": "disabled", "pooled_frequency": False}
COMBO_OPTIONS = {"compilation": "decoderFeatures", "pooled_frequency": True}
PHASES = ["short-4s", "continuous-32s"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def invalid(value):
        raise ValueError(f"Nonstandard JSON number: {value}")
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def canonical_id(value):
    require(isinstance(value, str), "Missing run UUID")
    parsed = uuid.UUID(value)
    require(str(parsed) == value.lower(), "Noncanonical run UUID")
    return str(parsed)


@dataclass(frozen=True)
class Variant:
    name: str
    compilation: str
    pooling: bool

    @property
    def options(self):
        return {"compilation": self.compilation, "pooled_frequency": self.pooling}

    def arguments(self, short):
        result = ["--candidate-compilation", self.compilation,
                  "--candidate-pooling", str(self.pooling).lower()]
        return result + (["--short-only"] if short else [])


SCREENS = (Variant("compiled-only", "decoderFeatures", False),
           Variant("pool-only", "disabled", True), Variant("body-plus-pool", "body", True))


def load_identity(config_path):
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    require(config.get("schemaVersion") == 1 and config.get("layers") == 8, "Unexpected local configuration")
    require(config.get("baselineRevision") == BASELINE, "Local baseline must be pinned2446c83")
    require(config.get("baselineOptions") == {"compilation": "disabled", "pooledFrequency": False},
            "Unexpected configured baseline options")
    require(config.get("candidateOptions") == {"compilation": "decoderFeatures", "pooledFrequency": True},
            "Expected the configured combo candidate")
    revision = config.get("candidateRevision", "")
    require(revision.startswith("source-sha256:") and benchmark.valid_hash(revision[14:]),
            "Local candidate source must be frozen")
    source_manifest = read_json(config_path.with_name("candidate-source.json"))
    require(source_manifest.get("source_sha256") == revision[14:], "Source manifest/config mismatch")
    require(config.get("requireBitwiseOutputEquality") is True and config.get("paced") is False,
            "Expected bitwise parity and unpaced replay")
    require(config.get("checkpointSHA256") == benchmark.CHECKPOINT_SHA256, "Unexpected model checkpoint")
    require(config.get("phases") == [
        {"id": "short-4s", "repetitions": 1, "trialOrder": list(benchmark.ORDER)},
        {"id": "continuous-32s", "repetitions": 8, "trialOrder": list(benchmark.ORDER)}],
        "Unexpected configured ABBA plan")
    fixture = (config_path.parent / config["inputFile"]).resolve()
    require(fixture.parent == config_path.parent and fixture.is_file(), "Missing local input fixture")
    data = fixture.read_bytes()
    require(len(data) == 4 * 16000 * 4, "Expected the staged four-second float32 fixture")
    return {"candidate_revision": revision, "checkpoint_sha256": config["checkpointSHA256"],
            "input_sha256": hashlib.sha256(data).hexdigest(),
            "configuration_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "config_path": str(config_path)}


def check_common(report, identity, variant, short, run_id, strict_scope=True):
    require(isinstance(report, dict), "Report is not an object")
    require(canonical_id(report.get("run_id")) == canonical_id(run_id), "Stale or wrong report UUID")
    require(report.get("baseline_revision") == BASELINE, "Wrong baseline revision")
    for key in ("candidate_revision", "checkpoint_sha256", "input_sha256"):
        require(report.get(key) == identity[key], f"Wrong {key}")
    require(report.get("layers") == 8 and report.get("precision") == "float32"
            and report.get("lookahead_frames") == 0, "Wrong layers, precision or lookahead")
    require(report.get("baseline_options") == BASE_OPTIONS and report.get("candidate_options") == variant.options,
            "Wrong effective engine options")
    require(report.get("optimized_argument_for_both_variants") is True
            and report.get("paced_input") is False, "Wrong optimizer/replay configuration")
    require(report.get("require_bitwise_output_equality") is True
            and report.get("output_parity_requirement") == "bitwise_equal"
            and report.get("absolute_tolerance_is_acceptance_gate") is False, "Parity gate was changed")
    require(report.get("swift_compilation_mode") == "wholemodule", "Wrong compilation mode")
    device = report.get("device")
    require(isinstance(device, dict) and device.get("physical_device") is True
            and isinstance(device.get("hardware_model"), str)
            and isinstance(device.get("os_version"), str), "Missing physical-device identity")
    if strict_scope:
        require(report.get("short_only") is short and report.get("planned_phases") == (PHASES[:1] if short else PHASES),
                "Wrong effective phase plan")
        require(report.get("benchmark_scope") == ("short_only_ablation" if short else "full_abba"),
                "Wrong benchmark scope")
        require(report.get("configured_candidate_options") == COMBO_OPTIONS, "Configured candidate changed")
        require(report.get("runtime_override_arguments") == variant.arguments(short), "Runtime arguments mismatch")


class Events:
    def __init__(self, identity, variant, short, seen):
        self.identity, self.variant, self.short, self.seen = identity, variant, short, seen
        self.run_id, self.started, self.terminal = None, None, None
        self.rows = []

    def accept(self, event):
        require(isinstance(event, dict), "Invalid console event")
        require(event.get("event") in {"run_started", "trial_started", "trial_completed", "run_completed", "failed"},
                "Unknown benchmark event")
        run_id = canonical_id(event.get("run_id"))
        if self.run_id is None:
            require(event.get("event") == "run_started", "Terminal/trial event arrived before run_started")
            require(run_id not in self.seen, "Run UUID was reused")
            require(event.get("layers") == 8 and event.get("baseline_revision") == BASELINE,
                    "Wrong console baseline/layers")
            require(event.get("candidate_revision") == self.identity["candidate_revision"], "Wrong console source")
            require(event.get("baseline_options") == BASE_OPTIONS and event.get("candidate_options") == self.variant.options,
                    "Wrong console options")
            require(event.get("short_only") is self.short
                    and event.get("planned_phases") == (PHASES[:1] if self.short else PHASES),
                    "Wrong console phase plan")
            require(isinstance(event.get("hardware"), str) and event["hardware"], "Missing console hardware")
            self.run_id, self.started = run_id, event
            self.seen.add(run_id)
        else:
            require(run_id == self.run_id, "Mixed run UUIDs in launch console")
            require(self.terminal is None, "Event arrived after terminal event")
            require(event.get("event") != "run_started", "Repeated run_started event")
        self.rows.append(event)
        if event.get("event") in ("run_completed", "failed"):
            if event["event"] == "run_completed":
                require(event.get("trials") == (4 if self.short else 8)
                        and event.get("result_file") == "Documents/latest-results.json", "Wrong completion event")
            self.terminal = event


def launch_command(device, variant, short):
    return ["xcrun", "devicectl", "device", "process", "launch", "--device", device,
            "--terminate-existing", "--console", BUNDLE, "--autorun"] + variant.arguments(short)


def capture_launch(command, directory, events, timeout):
    write_json(directory / "launch-command.json", command)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + timeout
    pending = b""
    try:
        with (directory / "console.log").open("wb") as console, (directory / "events.jsonl").open("w") as event_file:
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            while events.terminal is None:
                require(time.monotonic() < deadline, "Benchmark console timeout; no retry will be attempted")
                ready = selector.select(min(1.0, max(0.0, deadline - time.monotonic())))
                if not ready:
                    require(process.poll() is None, "Console exited before a matching terminal event")
                    continue
                chunk = os.read(process.stdout.fileno(), 65536)
                require(bool(chunk), "Console closed before a matching terminal event")
                console.write(chunk)
                console.flush()
                pending += chunk
                require(len(pending) <= 1024 * 1024, "Oversized unterminated console line")
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    text = line.decode("utf-8", errors="replace")
                    if MARKER in text:
                        event = json.loads(text.split(MARKER, 1)[1])
                        event_file.write(json.dumps(event, sort_keys=True) + "\n")
                        event_file.flush()
                        events.accept(event)
                        print(f"{events.variant.name}: {event.get('event')} {events.run_id}", flush=True)
    finally:
        if process.poll() is None:
            process.terminate()  # Only our local console process, never another application.
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)


def copy_report(device, directory, name):
    destination = directory / name
    require(not destination.exists(), "Refusing to overwrite copied evidence")
    command = ["xcrun", "devicectl", "device", "copy", "from", "--device", device,
               "--domain-type", "appDataContainer", "--domain-identifier", BUNDLE,
               "--source", "Documents/" + name, "--destination", str(destination)]
    write_json(directory / (name + ".copy-command.json"), command)
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
    (directory / (name + ".copy.log")).write_bytes(completed.stdout)
    require(completed.returncode == 0 and destination.is_file(), f"Could not copy fresh {name}")
    return read_json(destination)


def nominal(summary):
    return summary["device"].get("low_power_mode") is False and all(
        row["thermal_before"] == "nominal" and row["thermal_after"] == "nominal"
        and bool(row["thermal_push_counts"]) and set(row["thermal_push_counts"]) == {"nominal"}
        for row in summary["trials"])


def ratio(summary, phase):
    variants = summary["phases"][phase]["variants"]
    return variants["candidate"]["pooled_latency"]["mean_ms"] / variants["baseline"]["pooled_latency"]["mean_ms"]


def validate_parity_failure(report, events, identity, variant, short):
    check_common(report, identity, variant, short, events.run_id)
    require(report.get("status") == "parity_failed", "Failure lacks a fresh parity_failed progress report")
    rows = report.get("trials")
    require(isinstance(rows, list) and rows and len(rows) <= (4 if short else 8), "Invalid partial parity evidence")
    expected = [(p, i, v) for p in (PHASES[:1] if short else PHASES) for i, v in enumerate(benchmark.ORDER)]
    require([(r.get("phase"), r.get("order_index"), r.get("variant")) for r in rows] == expected[:len(rows)],
            "Partial trial order mismatch")
    for index, row in enumerate(rows):
        require(canonical_id(row.get("run_id")) == events.run_id, "Wrong partial-trial UUID")
        require(row.get("engine_options") == (BASE_OPTIONS if row["variant"] == "baseline" else variant.options),
                "Wrong partial-trial options")
        parity = row.get("parity", {})
        require(parity.get("gate") == "bitwise_equal", "Partial trial used a different parity gate")
        require(parity.get("passed") is (index < len(rows) - 1), "Invalid partial parity sequence")
        different = parity.get("different_float_bit_patterns")
        require(isinstance(different, int) and not isinstance(different, bool)
                and 0 <= different <= row.get("output_samples", -1), "Invalid partial bit difference count")
        require(parity.get("bitwise_equal") is (different == 0)
                and parity.get("passed") is (different == 0), "Inconsistent partial parity")
        seconds = benchmark.PHASES[row["phase"]]
        raw = row.get("raw_frame_ms")
        require(row.get("all_finite") is True and row.get("output_samples") == seconds * 16000
                and row.get("audio_seconds") == seconds and benchmark.valid_hash(row.get("output_sha256")),
                "Invalid partial output evidence")
        require(isinstance(raw, list) and len(raw) == seconds * 50 + 1
                and all(benchmark.finite_number(v) and v >= 0 for v in raw), "Invalid partial timing evidence")
        require(benchmark.finite_number(parity.get("max_abs_difference"))
                and parity["max_abs_difference"] >= 0, "Invalid partial output difference")
    last = rows[-1]
    require(last["parity"].get("bitwise_equal") is False
            and isinstance(last["parity"].get("different_float_bit_patterns"), int)
            and last["parity"]["different_float_bit_patterns"] > 0, "Missing actual differing output bits")
    require(any(e.get("event") == "trial_completed" and e.get("trial") == last.get("id")
                and e.get("parity_passed") is False and e.get("different_float_bit_patterns", 0) > 0
                for e in events.rows), "No matching live parity-failure event")
    require(events.terminal.get("event") == "failed"
            and "Bitwise output parity failed" in events.terminal.get("error", ""), "Not a strict parity failure")
    check_trial_events(report, events)


def check_trial_events(report, events):
    rows = report["trials"]
    completed = [e for e in events.rows if e.get("event") == "trial_completed"]
    started = [e for e in events.rows if e.get("event") == "trial_started"]
    require([e.get("trial") for e in completed] == [r.get("id") for r in rows]
            and [e.get("trial") for e in started] == [r.get("id") for r in rows],
            "Live trial events differ from copied report")
    for event, row in zip(completed, rows):
        parity = row["parity"]
        require(event.get("parity_passed") is parity["passed"]
                and event.get("different_float_bit_patterns") == parity["different_float_bit_patterns"]
                and event.get("max_abs_difference") == parity["max_abs_difference"],
                "Live parity differs from copied report")


def best_screen(runs, reference_ratio):
    eligible = [r for r in runs if r.get("status") == "validated" and r.get("timing_eligible")
                and r["short_ratio"] <= reference_ratio * 0.98]
    return min(eligible, key=lambda r: r["short_ratio"]) if eligible else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, help="Explicit intended CoreDevice identifier/UDID; never auto-selected")
    parser.add_argument("--output", required=True, type=Path, help="New or empty directory for all launch evidence")
    parser.add_argument("--reference-report", type=Path, help="Current combo full raw report; its SHORT ratio screens variants")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "BenchAssets/config.json")
    parser.add_argument("--timeout", type=float, default=600, help="Finite seconds per launch (default600); no retry")
    args = parser.parse_args()
    require(args.device.strip() == args.device and args.device and not args.device.startswith("-")
            and not any(ord(c) < 32 for c in args.device), "Invalid explicit device identifier")
    require(math.isfinite(args.timeout) and 0 < args.timeout <= 1200, "Timeout must be 1..1200 seconds")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    require(not any(output.iterdir()), "Output directory must be empty; evidence is never reused")
    manifest = {"schema_version": 1, "status": "preflight", "device_argument": args.device,
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "maximum_launches": 4, "runs": [], "selection_rule": {
                    "screen_phase": "short-4s", "reference_phase": "short-4s",
                    "metric": "candidate/baseline pooled mean frame latency, all frames retained",
                    "threshold": "screen_ratio <= reference_short_ratio * 0.98",
                    "selection": "one lowest eligible ratio; at most one full4s+32s confirmation",
                    "confirmation": "continuous32s confirmation is authoritative; short screens are provisional",
                    "timing_eligibility": "physical device, low-power mode off, every thermal observation nominal"}}
    def persist():
        write_json(output / "ablation.json", manifest)
    persist()
    seen, hardware, reference_ratio = set(), None, None
    try:
        identity = load_identity(args.config)
        manifest["identity"] = identity
        write_json(output / "config-snapshot.json", read_json(args.config))
        write_json(output / "candidate-source-snapshot.json", read_json(args.config.with_name("candidate-source.json")))
        if args.reference_report:
            reference = read_json(args.reference_report)
            check_common(reference, identity, Variant("combo", "decoderFeatures", True), False,
                         reference.get("run_id"), strict_scope=False)
            reference_summary = benchmark.summarize(reference)
            require(not reference_summary["short_only"] and nominal(reference_summary), "Reference is not a nominal full combo run")
            hardware = (reference["device"]["hardware_model"], reference["device"]["os_version"])
            reference_ratio = ratio(reference_summary, "short-4s")
            manifest["reference"] = {"path": str(args.reference_report.resolve()), "run_id": reference["run_id"],
                                     "short_ratio": reference_ratio, "continuous_ratio": ratio(reference_summary, "continuous-32s")}
            write_json(output / "reference-results.json", reference)
            write_json(output / "reference-summary.json", reference_summary)
        manifest["status"] = "running"
        persist()

        def run_one(variant, short):
            nonlocal hardware
            require(len(manifest["runs"]) < 4, "Maximum launch count reached")
            directory = output / f"{len(manifest['runs']) + 1:02d}-{variant.name}-{'short' if short else 'full'}"
            directory.mkdir()
            entry = {"variant": variant.name, "options": variant.options, "short_only": short,
                     "evidence_directory": directory.name, "status": "launching"}
            manifest["runs"].append(entry)
            persist()
            events = Events(identity, variant, short, seen)
            capture_launch(launch_command(args.device, variant, short), directory, events, args.timeout)
            entry["run_id"] = events.run_id
            entry["terminal_event"] = events.terminal
            name = "latest-results.json" if events.terminal["event"] == "run_completed" else "latest-progress.json"
            report = copy_report(args.device, directory, name)
            check_common(report, identity, variant, short, events.run_id)
            observed_hardware = (report["device"]["hardware_model"], report["device"]["os_version"])
            require(events.started["hardware"] == observed_hardware[0], "Console/report hardware mismatch")
            require(hardware is None or observed_hardware == hardware, "Device hardware/OS changed")
            hardware = observed_hardware
            if events.terminal["event"] == "failed":
                validate_parity_failure(report, events, identity, variant, short)
                entry.update(status="excluded_parity_failure", error=events.terminal["error"])
                print(f"Excluded {variant.name}: strict output parity failed; no retry", flush=True)
            else:
                summary = benchmark.summarize(report)
                check_trial_events(report, events)
                write_json(directory / "summary.json", summary)
                entry.update(status="validated", short_ratio=ratio(summary, "short-4s"), timing_eligible=nominal(summary))
                if not short:
                    entry["continuous_ratio"] = ratio(summary, "continuous-32s")
            persist()
            return entry

        for variant in SCREENS:
            run_one(variant, True)  # Each distinct screen runs once; no retry on any failure.
        selected = best_screen(manifest["runs"], reference_ratio) if reference_ratio is not None else None
        manifest["selection"] = {"variant": selected["variant"] if selected else None,
                                 "reference_short_ratio": reference_ratio,
                                 "reason": "threshold_met" if selected else "no_reference_or_no_eligible_screen_met_threshold"}
        persist()
        if selected:
            variant = next(v for v in SCREENS if v.name == selected["variant"])
            confirmation = run_one(variant, False)
            manifest["selection"]["confirmation_status"] = confirmation["status"]
        manifest["status"] = "completed"
        persist()
        print(str(output / "ablation.json"), flush=True)
        return 0
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
        manifest.update(status="stopped", error=str(error), no_retry=True)
        persist()
        print(f"Ablation stopped: {error}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        manifest.update(status="interrupted", no_retry=True)
        persist()
        return 130


if __name__ == "__main__":
    sys.exit(main())
