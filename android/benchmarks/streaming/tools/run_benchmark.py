#!/usr/bin/env python3
"""Run the nondebuggable release app on the owned emulator and verify its result."""
import argparse
import base64
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import queue
import subprocess
import threading
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ADB = Path(os.environ.get("ANDROID_HOME", str(Path.home() / "Library/Android/sdk"))) / "platform-tools/adb"
PACKAGE = "io.riftco.reusestreaming.benchmark"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", type=Path, default=DEFAULT_ADB)
    parser.add_argument("--serial", default="emulator-5554")
    parser.add_argument("--avd", default="ReuseAudioSdkTest")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--primary-only", action="store_true")
    parser.add_argument("--layers")
    parser.add_argument("--threads")
    parser.add_argument("--comparison-cases", help="Explicit native backend@threads entries joined by +")
    parser.add_argument("--capture-output", action="store_true")
    parser.add_argument("--input-asset")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    adb = [str(args.adb), "-s", args.serial]
    if not args.serial.startswith("emulator-"):
        raise ValueError("This harness run is authorized for an emulator")

    def call(*command):
        return subprocess.check_output(adb + list(command), text=True, timeout=30).strip()

    if call("shell", "getprop", "ro.kernel.qemu") != "1" or call("emu", "avd", "name").splitlines()[0].strip() != args.avd:
        raise ValueError("The connected emulator is not the owned benchmark AVD")
    if not 100 <= args.frames <= 1500:
        raise ValueError("Choose 100–1500 measured hops")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    run_id = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d-%H%M%S")
    environment = {
        "runId": run_id, "serial": args.serial, "avd": args.avd,
        "measuredFramesRequested": args.frames,
        "host": {"uname": subprocess.check_output(["uname", "-srm"], text=True).strip(),
                 "cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
                 "memoryBytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)),
                 "logicalCores": int(subprocess.check_output(["sysctl", "-n", "hw.logicalcpu"], text=True)),
                 "loadBefore": os.getloadavg()},
        "guest": {"fingerprint": call("shell", "getprop", "ro.build.fingerprint"),
                  "api": call("shell", "getprop", "ro.build.version.sdk"),
                  "abi": call("shell", "getprop", "ro.product.cpu.abi"),
                  "onlineCores": call("shell", "getconf", "_NPROCESSORS_ONLN"),
                  "meminfo": call("shell", "cat", "/proc/meminfo")},
        "scope": "Unpaced synthetic FP32 CPU processing on an emulator; not physical-phone or call latency",
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    call("shell", "am", "force-stop", PACKAGE)
    logcat = subprocess.Popen(adb + ["logcat", "-v", "raw", "-T", "1", "REUSE_BENCH:I", "*:S"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    launch = ["shell", "am", "start", "-W", "-n", PACKAGE + "/.MainActivity",
              "--es", "runId", run_id, "--ei", "timedFrames", str(args.frames)]
    if args.primary_only:
        launch += ["--ez", "primaryOnly", "true"]
    if args.comparison_cases:
        entry = r"(?:frequency-v1|bulk-v2|inline-v2|stable-v2)@[1-8]"
        if not re.fullmatch(entry + r"(?:\+" + entry + r"){0,7}", args.comparison_cases):
            raise ValueError("Invalid native comparison case sequence")
        launch += ["--es", "comparisonCases", args.comparison_cases]
    if args.capture_output:
        launch += ["--ez", "captureOutput", "true"]
    if args.input_asset:
        if not re.fullmatch(r"[A-Za-z0-9_./-]+", args.input_asset) or ".." in args.input_asset.split("/"):
            raise ValueError("Invalid validation input asset")
        launch += ["--es", "inputAsset", args.input_asset]
    for name, value in (("layers", args.layers), ("threads", args.threads)):
        if value:
            if not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", value):
                raise ValueError("Invalid case selection")
            launch += ["--es", name, value]
    lines = queue.Queue()
    def read_logcat():
        for line in logcat.stdout:
            lines.put(line)
    reader = threading.Thread(target=read_logcat, daemon=True)
    reader.start()
    chunks, count, active = {}, None, False
    deadline = time.monotonic() + args.timeout_seconds
    try:
        print(call(*launch), flush=True)
        with (output / "logcat.txt").open("w") as log:
            while time.monotonic() < deadline:
                try:
                    line = lines.get(timeout=1)
                except queue.Empty:
                    if logcat.poll() is not None:
                        raise RuntimeError("ADB logcat exited before a verified benchmark result")
                    continue
                if not line:
                    continue
                log.write(line)
                log.flush()
                text = line.strip()
                if text == "RUN_START " + run_id:
                    active = True
                    print(text, flush=True)
                if not active:
                    continue
                if text.startswith(("PROGRESS", "FAILURE", "FAILED")):
                    print(text, flush=True)
                chunk = re.fullmatch(r"RESULT_CHUNK (\d+)/(\d+):([A-Za-z0-9+/=]+)", text)
                if chunk:
                    index, total = int(chunk[1]), int(chunk[2])
                    if count not in (None, total) or not 1 <= index <= total or index in chunks:
                        raise RuntimeError("Invalid/duplicate benchmark result chunk")
                    chunks[index], count = chunk[3], total
                sha = re.fullmatch(r"RESULT_SHA256:([0-9a-f]{64})", text)
                if sha:
                    if count is None or set(chunks) != set(range(1, count + 1)):
                        raise RuntimeError("Missing benchmark result chunks")
                    raw = base64.b64decode("".join(chunks[i] for i in range(1, count + 1)), validate=True)
                    if hashlib.sha256(raw).hexdigest() != sha[1]:
                        raise RuntimeError("Benchmark transport SHA-256 mismatch")
                    data = json.loads(raw)
                    if data.get("runId") != run_id:
                        raise RuntimeError("Benchmark result belongs to another run")
                    (output / "results.json").write_bytes(raw)
                    environment["host"]["loadAfter"] = os.getloadavg()
                    environment["resultSha256"] = sha[1]
                    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
                    print("Verified result saved:", output / "results.json", flush=True)
                    if data.get("success") is not True:
                        raise RuntimeError("Benchmark app reported failure; inspect the saved partial result")
                    if args.capture_output:
                        pcm = output / "pcm"
                        pcm.mkdir()
                        for case in data["cases"]:
                            for kind in ("input", "output"):
                                record = case["export"][kind]
                                remote = record["file"]
                                prefix = "/Android/data/" + PACKAGE + "/files/benchmark-export/" + run_id + "-"
                                if prefix not in remote or not remote.endswith("-" + kind + ".f32") or ".." in remote.split("/"):
                                    raise RuntimeError("Unexpected benchmark-owned PCM export path")
                                destination = pcm / Path(remote).name
                                subprocess.run(adb + ["pull", remote, str(destination)], check=True, timeout=30)
                                raw_pcm = destination.read_bytes()
                                if len(raw_pcm) != record["bytes"] or hashlib.sha256(raw_pcm).hexdigest() != record["sha256"]:
                                    raise RuntimeError("Exported PCM integrity failure")
                        print("Verified raw input/output PCM exports:", pcm, flush=True)
                    return
        raise TimeoutError("No complete benchmark result before the bounded deadline; inspect logcat")
    finally:
        logcat.terminate()
        try:
            logcat.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logcat.kill()
            logcat.wait(timeout=5)
        reader.join(timeout=5)


if __name__ == "__main__":
    main()
