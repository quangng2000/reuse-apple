"""Real stateful ORT vs pinned upstream CPU reference and synthetic audio fixtures."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
from common import GENERATED, DEFAULT_CHECKPOINT, CHECKPOINT_SHA256, selected_weights
from oracle import Oracle


def metric(expected, actual):
    assert expected.shape == actual.shape and np.isfinite(expected).all() and np.isfinite(actual).all()
    diff = expected.astype(np.float64) - actual.astype(np.float64)
    return {"maxAbs": float(abs(diff).max()), "rmse": float(np.sqrt(np.mean(diff * diff)))}


def write(directory, name, data):
    path = directory / name
    np.asarray(data, dtype="<f4").tofile(path)
    return {"file": name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "shape": list(data.shape)}


def analyze_audio(samples):
    # Same mathematical centered reflection and EOF ceil-hop flush as the API.
    # This uses the independent PyTorch FFT, not Kotlin/native or ONNX transforms.
    count = len(samples)
    assert count >= 640
    padded = np.concatenate((samples[1:321][::-1], samples, samples[:-1][::-1]))
    frames = 1 + (count + 319) // 320
    framed = torch.stack([torch.tensor(padded[t*320:t*320+640].copy()) for t in range(frames)])
    spec = torch.fft.rfft(framed * torch.hann_window(640))
    return torch.log1p(spec.abs()).numpy()[:, None, :, None], torch.angle(spec).numpy()[:, None, :, None]


def synthesize_audio(magnitudes, phases, count):
    win = torch.hann_window(640).numpy()
    overlap, envelope, output = np.zeros(640, np.float32), np.zeros(640, np.float32), []
    for index, (mag, pha) in enumerate(zip(magnitudes, phases)):
        mag = torch.expm1(torch.clamp_min(torch.tensor(mag.reshape(321)), 0))
        if (mag == 0).float().mean() > .5:
            mag[:] = 0
        pha = torch.tensor(pha.reshape(321))
        spectrum = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
        samples = torch.fft.irfft(spectrum, n=640).numpy()
        overlap += samples * win
        envelope += win * win
        if index:
            output.append((overlap[:320] / np.maximum(envelope[:320], 1e-12)).copy())
        overlap[:-320] = overlap[320:]; overlap[-320:] = 0
        envelope[:-320] = envelope[320:]; envelope[-320:] = 0
    result = np.concatenate(output)[:count]
    assert len(result) == count and np.isfinite(result).all()
    return result


def chunks(count):
    result, used, sizes = [], 0, [1, 319, 320, 127, 640, 17]
    while used < count:
        size = min(sizes[len(result) % len(sizes)], count - used)
        result.append(size); used += size
    return result


def run(checkpoint, manifest_path, directory, spectral_frames=3, audio_counts=(640,)):
    directory.mkdir(parents=True, exist_ok=True)
    info = json.loads(manifest_path.read_text())
    model = manifest_path.with_name(info['modelFile'])
    assert hashlib.sha256(model.read_bytes()).hexdigest() == info['modelSha256']
    _, selected = selected_weights(checkpoint, info['layers'])
    graph = onnx.load(model)
    initializers = {n.name: onnx.numpy_helper.to_array(n) for n in graph.graph.initializer}
    assert all(initializers[k].dtype == np.float32 and initializers[k].tobytes() == w.numpy().tobytes() for k, w in selected.items())
    assert sum(n.op_type == 'Scan' for n in graph.graph.node) == info['layers'] * 2
    options = ort.SessionOptions(); options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(model), options, providers=['CPUExecutionProvider'])
    oracle = Oracle(checkpoint, info['layers'])
    state_specs = info['states']
    def zeros(): return {s['inputName']: np.zeros(s['shape'], np.float32) for s in state_specs}
    def step(mag, pha, state):
        output = dict(zip(info['outputNames'], session.run(info['outputNames'], dict(magnitude=mag, phase=pha, **state))))
        next_state = {s['inputName']: output[s['outputName']] for s in state_specs}
        assert all(v.dtype == np.float32 and v.shape == tuple(s['shape']) and np.isfinite(v).all()
                   for s in state_specs for v in [next_state[s['inputName']]])
        return output['enhanced_magnitude'], output['enhanced_phase'], next_state
    rng = np.random.default_rng(241024)
    mag = rng.uniform(0, 1, (spectral_frames, 1, 321, 1)).astype(np.float32)
    pha = rng.uniform(-np.pi, np.pi, mag.shape).astype(np.float32)
    state, expected_mag, expected_pha, actual_mag, actual_pha = zeros(), [], [], [], []
    max_state_error = 0.0
    first = None
    started = time.monotonic()
    for index in range(spectral_frames):
        rm, rp = oracle.step(mag[index], pha[index])
        am, ap, state = step(mag[index], pha[index], state)
        if first is None: first = (am.copy(), ap.copy())
        expected_mag.append(rm); expected_pha.append(rp); actual_mag.append(am); actual_pha.append(ap)
        cpu_states = oracle.states()
        for s in state_specs:
            a, b = cpu_states[s['key']], state[s['inputName']]
            max_state_error = max(max_state_error, metric(a, b)['maxAbs'])
            if not np.allclose(a, b, atol=3e-4, rtol=2e-4):
                raise AssertionError((s['key'], metric(a, b)))
    em, ep, am, ap = [np.stack(x) for x in [expected_mag, expected_pha, actual_mag, actual_pha]]
    mag_error = metric(em, am)
    wrapped_phase = np.angle(np.exp(1j * (ep.astype(np.float64) - ap)))
    phase_error = metric(np.zeros_like(wrapped_phase), wrapped_phase)
    assert mag_error['maxAbs'] < .003
    reset_m, reset_p, _ = step(mag[0], pha[0], zeros())
    assert np.array_equal(first[0], reset_m) and np.array_equal(first[1], reset_p)
    fresh_m, _, _ = step(mag[-1], pha[-1], zeros())
    carry_effect = metric(am[-1], fresh_m)['maxAbs']
    assert carry_effect > 1e-6, 'State must affect subsequent hops'
    files = {}
    def record(name, data):
        files[name] = write(directory, name, data)
        return name
    spectral = {"frames": spectral_frames, "magnitudeFile": record('spectral-magnitude.f32', mag),
                "phaseFile": record('spectral-phase.f32', pha),
                "enhancedMagnitudeFile": record('spectral-enhanced-magnitude.f32', em),
                "enhancedPhaseFile": record('spectral-enhanced-phase.f32', ep)}
    receipt = {'schemaVersion': 1, 'layers': info['layers'], 'modelSha256': info['modelSha256'],
               'checkpointSha256': CHECKPOINT_SHA256, 'selectedWeightBytesUnchanged': True,
               'source': 'hash-verified original NVIDIA model and StreamingSEMamba, eager CPU Mamba adapter',
               'dtype': 'float32', 'providers': session.get_providers(), 'spectral': spectral,
               'spectralMagnitudeError': mag_error, 'spectralWrappedPhaseError': phase_error,
               'maximumStateAbsError': max_state_error, 'stateAtol': .0003, 'stateRtol': .0002,
               'resetBitwiseEqual': True, 'carriedStateOutputDifferenceFromFresh': carry_effect,
               'waveformMaxAbsThreshold': .0003, 'fixtures': [], 'files': files,
               'scope': 'synthetic stateful numerical validation, not Android latency or broad audio quality'}
    for count in audio_counts:
        t = np.arange(count, dtype=np.float64) / 16000
        x = (.17*np.sin(2*np.pi*173*t) + .08*np.sin(2*np.pi*917*t) +
             .013*np.random.default_rng(421).standard_normal(count)).astype(np.float32)
        im, ip = analyze_audio(x)
        oracle.reset(); state = zeros(); expected, actual = ([], []), ([], [])
        for frame in range(len(im)):
            rm, rp = oracle.step(im[frame], ip[frame]); expected[0].append(rm); expected[1].append(rp)
            rm, rp, state = step(im[frame], ip[frame], state); actual[0].append(rm); actual[1].append(rp)
        reference_wave = synthesize_audio(*expected, count)
        actual_wave = synthesize_audio(*actual, count)
        errors = metric(reference_wave, actual_wave)
        assert errors['maxAbs'] <= .0003, errors
        fixture = {'name': f'synthetic-{count}', 'samples': count, 'chunks': chunks(count),
                   'inputFile': record(f'{count}-input.f32', x),
                   'outputFile': record(f'{count}-output.f32', reference_wave), 'waveformError': errors}
        receipt['fixtures'].append(fixture)
    receipt['elapsedSeconds'] = time.monotonic() - started
    receipt['passed'] = True
    (directory / 'parity.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ['layers','spectralMagnitudeError','spectralWrappedPhaseError',
                                            'maximumStateAbsError','resetBitwiseEqual','elapsedSeconds','passed']}), flush=True)
    del oracle, session
    gc.collect()
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--layers', type=int, nargs='+', default=[4,8])
    parser.add_argument('--models-directory', type=Path, default=GENERATED/'models')
    parser.add_argument('--output-directory', type=Path, default=GENERATED/'parity')
    parser.add_argument('--short', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    for layer in args.layers:
        extended = layer in [4,8] and not args.short
        run(args.checkpoint, args.models_directory/f'reuse-streaming-l{layer}-fp32.manifest.json',
            args.output_directory/f'l{layer}', 8 if extended else 3,
            (640,641,959,960,1377,8000,16000) if extended else (640,))


if __name__ == '__main__': main()
