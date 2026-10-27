#!/usr/bin/env python3
"""Independently check the one-phone Vulkan experiment and its exported waveform quality."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'android/benchmarks/streaming/tools'))
from compare_quality import reference_metrics

def sha(data): return hashlib.sha256(data).hexdigest()
def read_checked(path):
    raw = path.read_bytes()
    sidecar = path.with_suffix('.sha256') if path.suffix == '.json' else Path(str(path)+'.sha256')
    assert sidecar.read_text().split()[0] == sha(raw), f'Integrity mismatch {path.name}'
    return raw

def verify(folder, run_id, expected_manifest, clean_file, require_phone=True):
    raw = read_checked(folder/(run_id+'-result.json')); result = json.loads(raw)
    assert result['schemaVersion'] == 1 and result['runId'] == run_id and result['success'] is True
    assert result['layers'] == 8 and result['precision'] == 'FP32' and result['sampleRate'] == 16000 and result['hopSamples'] == 320
    assert result['warmupHops'] == 32, 'Unexpected warmup policy'
    assert result['bundleManifestSha256'] == expected_manifest
    device = result['device']; vk = result['vulkan']
    if require_phone:
        assert not device['emulatorReported'] and device['model'] == 'SM-S938U1' and device['sdk'] == 35
        assert 'adreno' in vk['deviceName'].lower(), 'Expected physical Qualcomm Adreno GPU'
    assert vk['hardwareGpu'] is True and vk['softwareDeviceAllowed'] is False and vk['precision'] == 'float32'
    assert vk['stateCount'] == 31 and vk['statePingPong'] is True and vk['gpuFiniteChecks'] is True
    assert vk['arenaBytes'] == 104001280, 'Unexpected Vulkan arena/model allocation'
    assert vk['operations'] == 530 and not vk['tiledMatmulPipeline'] and not vk['tiledConvolutionPipeline']
    frames = result['timedFramesPerCase']; assert isinstance(frames, int) and not isinstance(frames, bool) and 32 <= frames <= 300
    if require_phone: assert frames == 300, 'The physical comparison requires 300 measured hops per case'
    assert len(result['cases']) == 4 and result['quality']['passed'] is True and result['qualityTiled']['passed'] is True
    assert result['vulkanTiled']['hardwareGpu'] is True and result['vulkanTiled']['softwareDeviceAllowed'] is False
    assert result['vulkanTiled']['deviceName'] == vk['deviceName']
    tiled = result['vulkanTiled']
    for key in ['vendorId','deviceId','driverVersion','apiVersion','precision','stateCount','statePingPong','gpuFiniteChecks','operations','arenaBytes']:
        assert tiled[key] == vk[key]
    assert tiled['tiledMatmulPipeline'] and tiled['tiledConvolutionPipeline']
    benchmark = read_checked(folder/(run_id+'-benchmark-input.f32'))
    assert len(benchmark) == (640+(32+frames)*320)*4 and np.isfinite(np.frombuffer(benchmark,dtype='<f4')).all()
    assert result['benchmarkInputSha256'] == sha(benchmark)
    full = (REPO/'Validation/AndroidStreamingOptimization-vulkan/benchmark-input.f32').read_bytes()
    assert sha(full) == 'e3150c5d89ff157b323627de4e5ede75a35be2e44088b284daa9d47268ccb9ca'
    assert full[:len(benchmark)] == benchmark
    for case, label, backend, threads in zip(result['cases'],
            ['stable-v2-cpu-before','vulkan-gpu-direct','vulkan-gpu-tiled','stable-v2-cpu-after'],
            ['CPUExecutionProvider','Vulkan','Vulkan','CPUExecutionProvider'],[8,0,0,8]):
        assert case['caseId'] == label and case['backend'] == backend and case['cpuThreads'] == threads
        assert case['layers'] == 8 and case['precision'] == 'FP32' and case['frames'] == frames and case['allOutputFinite'] is True
        assert case['inputSamples'] == case['outputSamples'] == 640+(32+frames)*320
        times = np.asarray(case['rawPushMs']); render = np.asarray(case['rawRenderMs'])
        assert len(times) == len(render) == frames and np.isfinite(times).all() and np.isfinite(render).all()
        assert (times > 0).all() and (render > 0).all() and math.isfinite(case['checksum'])
        s = np.sort(times)
        expected = {'mean':times.mean(),'min':s[0],'max':s[-1]}
        expected.update({name:s[math.ceil(p*frames)-1] for name,p in [('p50',.5),('p95',.95),('p99',.99)]})
        for key, value in expected.items(): assert abs(case['latencyMs'][key]-value) < 1e-6
        assert case['over20Ms'] == int((times>20).sum()) and abs(case['rtf']-times.mean()/20) < 1e-6
    clean_raw = clean_file.read_bytes()
    assert sha(clean_raw) == 'd06618f56328344db958ec0ef058e27896c3f50389265ce4ce209cfc99ca2a40'
    clean = np.frombuffer(clean_raw,dtype='<f4').astype(np.float64)
    qualities = {}
    for variant, field in [('direct','quality'),('tiled','qualityTiled')]:
        waves = {}
        for name in ['input','cpu','vulkan']:
            data = read_checked(folder/(run_id+'-quality-'+variant+'-'+name+'.f32'))
            waves[name] = np.frombuffer(data,dtype='<f4').astype(np.float64)
            assert len(waves[name]) == 64000 and np.isfinite(waves[name]).all()
            if name == 'input': assert sha(data) == 'c449b1ebf8f51c80f7530961b41c62a733a1033a60140fc16c0b37186120835f'
        a,b = waves['cpu'],waves['vulkan']; error = b-a
        quality = {'maxAbs':float(np.abs(error).max()), 'relativeRms':float(np.sqrt(np.sum(error*error)/np.sum(a*a))),
               'cpuMetrics':reference_metrics(clean,a),'vulkanMetrics':reference_metrics(clean,b),
               'scope':'One four-second synthetic noisy speech sample; no alignment, scaling or cropping'}
        quality['stoiDelta'] = quality['vulkanMetrics']['stoi']-quality['cpuMetrics']['stoi']
        quality['siSdrDeltaDb'] = quality['vulkanMetrics']['siSdrDb']-quality['cpuMetrics']['siSdrDb']
        assert quality['maxAbs'] <= 5e-6 and quality['relativeRms'] <= 1e-5
        assert abs(quality['stoiDelta']) <= 1e-4 and abs(quality['siSdrDeltaDb']) <= .001
        assert abs(result[field]['maxAbs']-quality['maxAbs']) < 1e-12
        assert abs(result[field]['relativeRms']-quality['relativeRms']) < 1e-12
        assert result[field]['resetBitwiseReplay'] is True and result[field]['partialTailsAndCancellation'] is True
        qualities[variant] = quality
    best = min(result['cases'][1:3],key=lambda c:c['latencyMs']['mean'])
    return {'verified':True, 'resultSha256':sha(raw),'result':result,'independentQuality':qualities,
            'fastestGpuCase':best['caseId'],'under20Mean':best['latencyMs']['mean'] < 20,
            'under20EveryHop':best['latencyMs']['max'] < 20}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--expected-manifest',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--clean',type=Path,default=REPO/'Validation/AndroidStreamingOptimization-0.1.2/quality/clean-speech.f32')
    p.add_argument('--allow-emulator',action='store_true')
    a=p.parse_args(); result=verify(a.folder,a.run_id,a.expected_manifest,a.clean,not a.allow_emulator)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'verified':True,'under20Mean':result['under20Mean'],'under20EveryHop':result['under20EveryHop'],
          'latencies':[{k:c[k] for k in ['caseId','latencyMs','over20Ms']} for c in result['result']['cases']],
          'quality':result['independentQuality']},indent=2))

if __name__ == '__main__':main()
