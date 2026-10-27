"""Matched warmed baseline/native graph timings. Diagnostic only, not Android speed."""
import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np

from validate import Runner, session
from transform import SOURCE_SHA, digest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def stats(values):
    return {"hops": len(values), "meanMs": float(np.mean(values)), "medianMs": float(np.median(values)),
            "p95Ms": float(np.percentile(values, 95)), "maxMs": float(np.max(values))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hops", type=int, default=40)
    parser.add_argument("--output", type=Path, default=HERE / "generated/host-timing.json")
    args = parser.parse_args()
    base = ROOT / "tools/streaming-export/generated/models/reuse-streaming-l8-fp32.onnx"
    native = HERE / "generated/reuse-streaming-l8-native-frequency-v1.onnx"
    lib = HERE / ("generated/build-host/libreuse_streaming_scan" + (".dylib" if platform.system() == "Darwin" else ".so"))
    info = json.loads(native.with_suffix(".manifest.json").read_text())
    rng = np.random.default_rng(813)
    frames = [(rng.uniform(0, 1, (1, 321, 1)).astype(np.float32),
               rng.uniform(-np.pi, np.pi, (1, 321, 1)).astype(np.float32)) for _ in range(args.hops)]
    results = []
    for threads in (1, 2, 4):
        old, old_options = session(base, None, threads)
        new, new_options = session(native, lib, threads)
        runners = [Runner(old, info), Runner(new, info)]
        for engine in runners:
            for frame in frames[:5]:
                engine.step(*frame)
            engine.reset()
        times = [[], []]
        for hop, frame in enumerate(frames):
            # Alternate first/second execution to avoid a fixed ordering advantage.
            for index in ([0, 1] if hop % 2 == 0 else [1, 0]):
                runner = runners[index]
                inputs = dict(magnitude=frame[0], phase=frame[1], **runner.state)
                started = time.perf_counter_ns()
                values = runner.engine.run(info["outputNames"], inputs)
                times[index].append((time.perf_counter_ns() - started) / 1e6)
                runner.state = {spec["inputName"]: value for spec, value in zip(info["states"], values[2:])}
        results.append({"threads": threads, "baseline": stats(times[0]), "native": stats(times[1]),
                        "medianSpeedup": float(np.median(times[0]) / np.median(times[1]))})
    receipt = {"scope": "Mac host warmed Python ORT diagnostic; excludes Android/JNI/FFT and cannot prove phone latency",
               "platform": platform.platform(), "baselineModelSha256": SOURCE_SHA,
               "nativeModelSha256": digest(native), "nativeLibrarySha256": digest(lib),
               "results": results}
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
