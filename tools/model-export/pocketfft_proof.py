"""Prove the JNI's per-frame FP32 PocketFFT algorithm against pinned PyTorch.

Compiles an isolated host harness against the vendored header, not the Android
app. No downloads, model inference, audio recordings or SDK settings changes.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
import torch
from parity import error, signal

HEADER_SHA = "b15ca29cf47025e76c966d0c5a061e9b56e3e2b312cb05e993ee696f8d9812e5"
FLAGS = ["-std=c++17", "-O2", "-fno-fast-math", "-ffp-contract=on", "-DPOCKETFFT_NO_MULTITHREADING=1"]
CPP = r'''
#include "pocketfft_hdronly.h"
#include <fstream>
#include <vector>
#include <complex>
#include <stdexcept>
int main(int argc,char**argv) {
  if(argc!=4) return 2;
  std::ifstream file(argv[1],std::ios::binary|std::ios::ate);
  const auto size=file.tellg();
  if(size<=0 || size%(640*4)!=0) return 3;
  file.seekg(0);
  const size_t frames=size/(640*4);
  std::vector<float> x(frames*640), inverse(frames*640);
  std::vector<std::complex<float>> y(frames*321);
  file.read(reinterpret_cast<char*>(x.data()),size);
  for(size_t row=0;row<frames;++row) {
    pocketfft::r2c<float>({640},{4},{8},0,true,x.data()+row*640,y.data()+row*321,1.f,1);
    pocketfft::c2r<float>({640},{8},{4},0,false,y.data()+row*321,inverse.data()+row*640,1.f/640.f,1);
  }
  std::ofstream(argv[2],std::ios::binary).write(reinterpret_cast<char*>(y.data()),y.size()*8);
  std::ofstream(argv[3],std::ios::binary).write(reinterpret_cast<char*>(inverse.data()),inverse.size()*4);
}
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path(__file__).parent / "generated/pocketfft-proof")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    cpp = Path(__file__).resolve().parents[2] / "android/reuse-audio/src/main/cpp"
    header = cpp / "pocketfft_hdronly.h"
    assert hashlib.sha256(header.read_bytes()).hexdigest() == HEADER_SHA
    torch.set_num_threads(2)
    compiler = shutil.which("clang++") or shutil.which("c++")
    if not compiler:
        raise RuntimeError("C++17 host compiler required")
    harness = a.output / "proof.cpp"
    harness.write_text(CPP)
    binary = a.output / "proof"
    subprocess.run([compiler, *FLAGS, "-I", str(cpp), str(harness), "-o", str(binary)], check=True)
    receipt = {"headerSha256": HEADER_SHA, "flags": FLAGS, "torchVersion": torch.__version__,
               "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
               "scope": "host per-frame FFT proof only; Android native pipeline has separate tests", "cases": []}
    for count in (640, 8000, 64000):
        x = torch.tensor(signal(count))
        padded = torch.nn.functional.pad(x[None, None], (320, 320), mode="reflect")[0, 0]
        frames = (padded.unfold(0, 640, 80) * torch.hann_window(640)).contiguous()
        expected = torch.fft.rfft(frames).numpy()
        inverse = torch.fft.irfft(torch.tensor(expected), n=640).numpy()
        src, dest, inv = [a.output / f"{count}-{n}.f32" for n in ("frames", "fft", "inverse")]
        frames.numpy().astype("<f4").tofile(src)
        subprocess.run([str(binary.resolve()), str(src.resolve()), str(dest.resolve()), str(inv.resolve())], check=True)
        got = np.fromfile(dest, dtype="<c8").reshape(expected.shape)
        got_inverse = np.fromfile(inv, dtype="<f4").reshape(inverse.shape)
        complex_error = np.abs(expected - got)
        flips = int(np.count_nonzero(abs(np.angle(expected) - np.angle(got)) > np.pi))
        case = {"samples": count, "frames": len(frames), "forwardBitwiseEqual": bool(np.array_equal(expected, got)),
                "forwardComplexMaxAbs": float(complex_error.max()), "phaseBranchFlips": flips,
                "inverse": error(inverse, got_inverse)}
        receipt["cases"].append(case)
        if not case["forwardBitwiseEqual"] or flips or case["inverse"]["maxAbs"] > 1e-6:
            raise AssertionError(case)
    (a.output / "proof.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
