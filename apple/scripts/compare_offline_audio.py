#!/usr/bin/env python3
"""Compare raw, aligned FP32 outputs; requires numpy, soundfile and pystoi."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from pystoi import stoi


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['clean', 'input', 'reference', 'output-dir', 'report']:
        parser.add_argument('--' + key, type=Path, required=True)
    return parser.parse_args()


def load(path):
    samples, rate = sf.read(path, dtype='float64')
    if rate != 16000 or samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError(f'Expected finite 16 kHz mono audio: {path}')
    return samples


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def si_sdr(samples, clean):
    samples, clean = samples - samples.mean(), clean - clean.mean()
    if np.dot(clean, clean) == 0:
        raise ValueError('Clean reference must contain nonzero signal.')
    target = np.dot(samples, clean) / np.dot(clean, clean) * clean
    error = samples - target
    return float(10 * np.log10(max(np.dot(target, target), 1e-30) / max(np.dot(error, error), 1e-30)))


def compare(path, clean, reference):
    samples = load(path)
    if samples.shape != clean.shape:
        raise ValueError(f'Sample count differs: {path}')
    difference = samples - reference
    return dict(file=path.name, sha256=digest(path), samples=len(samples),
                stoi=float(stoi(clean, samples, 16000, extended=False)),
                si_sdr_db=si_sdr(samples, clean), peak=float(abs(samples).max()),
                max_abs_delta=float(abs(difference).max()),
                relative_rms_delta=float(np.linalg.norm(difference) / max(np.linalg.norm(reference), 1e-30)))


def main():
    options = arguments()
    clean, noisy, reference = map(load, [options.clean, options.input, options.reference])
    if clean.shape != noisy.shape or clean.shape != reference.shape:
        raise ValueError('Clean, input and reference must have identical sample counts.')
    files = sorted(options.output_dir.glob('*.wav'))
    if not files:
        raise ValueError('No output WAV files found.')
    result = dict(clean_sha256=digest(options.clean), input_sha256=digest(options.input),
                  reference_sha256=digest(options.reference), input_stoi=float(stoi(clean, noisy, 16000)),
                  input_si_sdr_db=si_sdr(noisy, clean), outputs=[compare(p, clean, reference) for p in files],
                  scope='Unmodified aligned samples: no gain matching, normalization, trimming, or resampling.')
    with options.report.open('x') as report:
        json.dump(result, report, indent=2, allow_nan=False)
        report.write('\n')


if __name__ == '__main__':
    main()
