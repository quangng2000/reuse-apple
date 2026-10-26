"""Compare a native raw-FP32 output with a recorded oracle, without inference."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from pystoi import stoi
from parity import error, si_sdr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--actual", required=True, type=Path)
    p.add_argument("--clean-wav", type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    reference = np.fromfile(a.reference, dtype="<f4")
    actual = np.fromfile(a.actual, dtype="<f4")
    metrics = error(reference, actual)
    receipt = {"referenceSha256": hashlib.sha256(a.reference.read_bytes()).hexdigest(),
               "actualSha256": hashlib.sha256(a.actual.read_bytes()).hexdigest(),
               "waveform": metrics, "waveformMaxAbsThreshold": .0003,
               "scope": "one explicitly supplied native-output diagnostic; not general quality or latency"}
    passed = metrics["maxAbs"] <= .0003
    if a.clean_wav:
        rate, clean = wavfile.read(a.clean_wav)
        if rate != 16000 or clean.shape != reference.shape or clean.dtype != np.float32:
            raise ValueError("Clean reference must be length-matched mono16k float32 WAV")
        q = {"referenceSTOI": float(stoi(clean, reference, 16000, extended=False)),
             "actualSTOI": float(stoi(clean, actual, 16000, extended=False)),
             "referenceSISDRDB": si_sdr(clean, reference), "actualSISDRDB": si_sdr(clean, actual),
             "stoiAbsDeltaThreshold": .0001, "siSDRAbsDeltaThresholdDB": .02}
        passed &= abs(q["referenceSTOI"] - q["actualSTOI"]) <= .0001
        passed &= abs(q["referenceSISDRDB"] - q["actualSISDRDB"]) <= .02
        receipt["pairedSpeechQuality"] = q
        receipt["cleanSha256"] = hashlib.sha256(a.clean_wav.read_bytes()).hexdigest()
    receipt["passed"] = bool(passed)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if not passed:
        raise AssertionError("Fixed native PCM waveform/quality gate failed")


if __name__ == "__main__":
    main()
