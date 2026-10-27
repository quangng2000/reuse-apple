#!/usr/bin/env python3
"""Verify the schema-2 same-phone Vulkan comparison from raw samples.

This checks model/bundle identity, synchronous timing, reset and waveform
numerical quality. Provider duration, billing and physical-device allocation
must additionally be checked against the Firebase execution receipt.
"""
import argparse
import json
import math
from pathlib import Path
import re

import numpy as np

from verify_phone_result import REPO, read_checked, reference_metrics, sha

BASELINE_MANIFEST = 'a2626151ecc5d1f1a5021e6a12d5bb0d691b03f70e5c3fe5a21c9c33f3a1a360'
GOLDEN_CPU_SHA = '54ea90bd246e3a527aa3c5ee00f9e8e9a2b10983b53153a6b1fe7427ca16e65f'
BENCHMARK_SHA = 'e3150c5d89ff157b323627de4e5ede75a35be2e44088b284daa9d47268ccb9ca'
NOISY_SHA = 'c449b1ebf8f51c80f7530961b41c62a733a1033a60140fc16c0b37186120835f'
CLEAN_SHA = 'd06618f56328344db958ec0ef058e27896c3f50389265ce4ce209cfc99ca2a40'


def number(value):
    assert isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    return value


def integer(value):
    assert isinstance(value, int) and not isinstance(value, bool)
    return value


def verify_gpu(vk, require_phone):
    assert vk['backend'] == 'native-cpp-vulkan-fp32'
    assert vk['hardwareGpu'] is True and vk['softwareDeviceAllowed'] is False
    assert vk['precision'] == 'float32' and integer(vk['stateCount']) == 31
    assert vk['statePingPong'] is True and vk['gpuFiniteChecks'] is True
    assert integer(vk['arenaBytes']) == 104001280 and integer(vk['operations']) == 530
    assert vk['tiledMatmulPipeline'] is True and vk['tiledConvolutionPipeline'] is True
    assert vk['perDispatchProfiling'] is False, 'Timing comparison must not include per-dispatch profiling'
    for key in ['vendorId', 'deviceId', 'deviceType', 'apiVersion', 'driverVersion', 'subgroupSize',
                'subgroupOperations', 'maxStorageBufferRange']:
        assert integer(vk[key]) > 0
    if require_phone:
        assert vk['deviceName'] == 'Adreno (TM) 830' and vk['vendorId'] == 20803
        assert vk['deviceId'] == 1141178369 and vk['subgroupSize'] == 64
    assert vk['maxStorageBufferRange'] >= vk['arenaBytes']
    assert type(vk['float32DenormPreserveSupported']) is bool


def verify(folder, run_id, expected_manifest, clean_file, require_phone=True, allow_structural_fixture=False):
    assert re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', run_id)
    assert re.fullmatch(r'[0-9a-f]{64}', expected_manifest) and expected_manifest != BASELINE_MANIFEST
    raw = read_checked(folder/(run_id+'-result.json')); result = json.loads(raw)
    assert allow_structural_fixture or '_structuralFixtureOnly' not in result, 'Adapted fixtures are not new benchmark evidence'
    assert integer(result['schemaVersion']) == 2 and result['runId'] == run_id and result['success'] is True
    assert integer(result['layers']) == 8 and result['precision'] == 'FP32'
    assert integer(result['sampleRate']) == 16000 and integer(result['hopSamples']) == 320
    assert integer(result['warmupHops']) == 32
    assert result['baselineBundleManifestSha256'] == BASELINE_MANIFEST
    assert result['bundleManifestSha256'] == expected_manifest
    assert 0 < integer(result['elapsedMs']) < 94000, 'Unexpected bounded-test elapsed time'
    device = result['device']; baseline = result['baselineVulkan']; optimized = result['vulkan']
    assert type(device['emulatorReported']) is bool
    if require_phone:
        assert device['emulatorReported'] is False and device['model'] == 'SM-S938U1'
        assert device['manufacturer'] == 'samsung' and integer(device['sdk']) == 35
        assert device['abis'] == ['arm64-v8a'] and device['fingerprint'].startswith('samsung/')
    for vk in [baseline, optimized]: verify_gpu(vk, require_phone)
    # Both variants must execute on the very same hardware and driver with
    # matching numerical and state semantics, not merely the same GPU name.
    for key in ['deviceName','vendorId','deviceId','deviceType','apiVersion','driverVersion',
                'subgroupSize','subgroupOperations','precision','stateCount','statePingPong',
                'gpuFiniteChecks','operations','arenaBytes','float32DenormPreserveSupported',
                'maxStorageBufferRange']:
        assert baseline[key] == optimized[key], f'Cross-variant device mismatch: {key}'
    frames = integer(result['timedFramesPerCase']); assert 32 <= frames <= 300
    if require_phone: assert frames == 300, 'Physical comparison requires 300 measured hops per case'
    assert len(result['cases']) == 4
    benchmark = read_checked(folder/(run_id+'-benchmark-input.f32'))
    count = 640+(32+frames)*320
    assert len(benchmark) == count*4 and np.isfinite(np.frombuffer(benchmark,dtype='<f4')).all()
    assert result['benchmarkInputSha256'] == sha(benchmark)
    full = (REPO/'Validation/AndroidStreamingOptimization-vulkan/benchmark-input.f32').read_bytes()
    assert sha(full) == BENCHMARK_SHA and full[:len(benchmark)] == benchmark
    labels = ['stable-v2-cpu-before','vulkan-tiled-baseline','vulkan-tiled-optimized','stable-v2-cpu-after']
    for case,label,backend,threads in zip(result['cases'], labels,
            ['CPUExecutionProvider','Vulkan','Vulkan','CPUExecutionProvider'],[8,0,0,8]):
        assert case['caseId'] == label and case['backend'] == backend and integer(case['cpuThreads']) == threads
        assert integer(case['layers']) == 8 and case['precision'] == 'FP32'
        assert integer(case['frames']) == frames and case['allOutputFinite'] is True
        assert integer(case['inputSamples']) == integer(case['outputSamples']) == count
        assert len(case['rawPushMs']) == len(case['rawRenderMs']) == frames
        times = np.asarray([number(x) for x in case['rawPushMs']], dtype=np.float64)
        renders = np.asarray([number(x) for x in case['rawRenderMs']], dtype=np.float64)
        assert (times > 0).all() and (renders > 0).all(); number(case['checksum'])
        s = np.sort(times)
        expected = {'mean':times.mean(),'min':s[0],'max':s[-1]}
        expected.update({name:s[math.ceil(p*frames)-1] for name,p in [('p50',.5),('p95',.95),('p99',.99)]})
        for key,value in expected.items(): assert abs(number(case['latencyMs'][key])-value) < 1e-6
        assert integer(case['over20Ms']) == int((times>20).sum())
        assert abs(number(case['rtf'])-times.mean()/20) < 1e-6
    clean_raw = clean_file.read_bytes(); assert sha(clean_raw) == CLEAN_SHA
    clean = np.frombuffer(clean_raw,dtype='<f4').astype(np.float64)
    waves = {}
    for name in ['input','cpu','vulkan']:
        data = read_checked(folder/(run_id+'-quality-optimized-'+name+'.f32'))
        waves[name] = np.frombuffer(data,dtype='<f4').astype(np.float64)
        assert len(waves[name]) == 64000 and np.isfinite(waves[name]).all()
        if name == 'input': assert sha(data) == NOISY_SHA
        if name == 'cpu': assert sha(data) == GOLDEN_CPU_SHA, 'Reference must be the pinned published-model output'
    a,b = waves['cpu'],waves['vulkan']; error = b-a
    quality = {'maxAbs':float(np.abs(error).max()),'relativeRms':float(np.sqrt(np.sum(error*error)/np.sum(a*a))),
               'cpuMetrics':reference_metrics(clean,a),'vulkanMetrics':reference_metrics(clean,b),
               'scope':'One four-second synthetic noisy speech sample against pinned published CPU output; no alignment, scaling or cropping'}
    quality['stoiDelta'] = quality['vulkanMetrics']['stoi']-quality['cpuMetrics']['stoi']
    quality['siSdrDeltaDb'] = quality['vulkanMetrics']['siSdrDb']-quality['cpuMetrics']['siSdrDb']
    assert quality['maxAbs'] <= 5e-6 and quality['relativeRms'] <= 1e-5
    assert abs(quality['stoiDelta']) <= 1e-4 and abs(quality['siSdrDeltaDb']) <= .001
    reported = result['quality']
    assert reported['passed'] is True and integer(reported['samples']) == 64000
    assert abs(number(reported['maxAbs'])-quality['maxAbs']) < 1e-12
    assert abs(number(reported['relativeRms'])-quality['relativeRms']) < 1e-12
    assert reported['resetBitwiseReplay'] is True and reported['partialTailsAndCancellation'] is True
    base_case,opt_case = result['cases'][1:3]
    baseline_mean = base_case['latencyMs']['mean']; mean = opt_case['latencyMs']['mean']
    return {'verified':True,'resultSha256':sha(raw),'result':result,'independentQuality':{'optimized':quality},
            'optimizedGpuCase':opt_case['caseId'],'under20Mean':mean < 20,
            'under20EveryHop':opt_case['latencyMs']['max'] < 20,
            'latencyReductionFraction':1-mean/baseline_mean,'gpuSpeedup':baseline_mean/mean,
            'cpuControlDriftFraction':result['cases'][3]['latencyMs']['mean']/result['cases'][0]['latencyMs']['mean']-1}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,required=True); p.add_argument('--run-id',required=True)
    p.add_argument('--expected-manifest',required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--clean',type=Path,default=REPO/'Validation/AndroidStreamingOptimization-0.1.2/quality/clean-speech.f32')
    p.add_argument('--allow-emulator',action='store_true')
    a = p.parse_args(); result = verify(a.folder,a.run_id,a.expected_manifest,a.clean,not a.allow_emulator)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['verified','under20Mean','under20EveryHop','latencyReductionFraction',
                                          'gpuSpeedup','cpuControlDriftFraction','independentQuality']},indent=2))


if __name__ == '__main__': main()
