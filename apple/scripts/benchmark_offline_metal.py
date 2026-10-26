#!/usr/bin/env python3
"""Sequential same-binary offline ablation; no model downloads or remote actions."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['binary', 'model', 'input', 'output-dir']:
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--modes', nargs='+', default=[
        'standard', 'scan-only', 'normalization-only', 'specialized-metal'])
    return parser.parse_args()


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def run(options, index, mode):
    output = options.output_dir / f'{index:02d}-{mode}.wav'
    command = [str(options.binary.resolve()), '--model', str(options.model.resolve()),
               '--input', str(options.input.resolve()), '--output', str(output.resolve()),
               '--optimization', mode, '--benchmark']
    with output.with_suffix('.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads(output.with_suffix('.json').read_text())
    times = result['timings_seconds']
    if len(times) != 3 or not all(math.isfinite(t) and t > 0 for t in times):
        raise ValueError('Expected three finite, positive inference timings.')
    record = dict(mode=mode, output=output.name, timings_seconds=times,
                  warmed_seconds=times[1:], thermal_states=result['thermal_states_before_after'],
                  samples=result['samples'], peak_bytes=result['mlx_peak_bytes'],
                  waveform_sha256=digest(output), report_sha256=digest(output.with_suffix('.json')))
    print(json.dumps({'mode': mode, 'warmed_mean_seconds': statistics.mean(times[1:])}), flush=True)
    return record


def main():
    options = arguments()
    if options.output_dir.exists() and any(options.output_dir.iterdir()):
        raise SystemExit('Output directory must be empty; previous evidence is never overwritten.')
    options.output_dir.mkdir(parents=True, exist_ok=True)
    modes = options.modes
    order = modes + list(reversed(modes))
    records = [run(options, index, mode) for index, mode in enumerate(order)]
    summary = {mode: statistics.mean(t for r in records if r['mode'] == mode
                                     for t in r['warmed_seconds']) for mode in modes}
    result = dict(binary_sha256=digest(options.binary), model_sha256=digest(options.model),
                  input_sha256=digest(options.input), records=records, warmed_means_seconds=summary,
                  scope='Full-file analysis + all 30 layers + synthesis, GPU completion; loading and file I/O excluded. First inference in each process excluded from warmed comparisons.')
    (options.output_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
