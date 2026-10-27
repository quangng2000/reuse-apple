#!/usr/bin/env python3
"""Verify schema-3 ABBA timing, exact package identity and independent FP32 waveform quality.

Billing, provider runtime and allocated physical device additionally require the
Firebase receipts. A passing local/emulator result is never physical evidence.
"""
import argparse
import json
import math
from pathlib import Path
import re
import struct
import numpy as np
from verify_phone_result import REPO, read_checked, reference_metrics, sha
from verify_optimized_result import number, integer, BENCHMARK_SHA, GOLDEN_CPU_SHA, NOISY_SHA, CLEAN_SHA

BASELINE_MANIFEST = '951adf47ebece2c27a5af4df22b91c30c8a442b8caa04bb01db102ceb0ac733a'
BASELINE_WEIGHTS_SHA = 'aabfcd6872419a8a529127a0ea180d422bed7343efe6228c9ba80f796f21e600'
CANDIDATE_MANIFEST_SHA = '744eb7fbf38da57ea04f37ce7de821037c28bc8a759820eb3098d2803a8c0d0d'
CANDIDATE_WEIGHTS_SHA = '58952f70dcef36db55a6e8a419d9aff0e412d082a1eef077d9217ded6652882d'
CANDIDATE_PLAN_SHA = '0b510c65465b1cfe9f7dcf83503a194a53c16c3b49367a786f308fe48f098cc4'
CANDIDATE_NATIVE_SHA = '507377ed71ff24cebcf1176995d91ee6ffc9b4d358e6cd6252b601c5baf2fcf5'
BASELINE_PLAN_SHA = '5e2d760d556a7d24d72b14aa7e3783b1d6ec0d9febf4b4fbe95a79ffc0628c27'
PHONE_IDENTITY = {'deviceName':'Adreno (TM) 830','vendorId':20803,'deviceId':1141178369,
                  'driverVersion':2150760459,'apiVersion':4206876,'subgroupSize':64,'deviceType':1}
IDENTITY_KEYS = tuple(PHONE_IDENTITY) + ('subgroupOperations','float32DenormPreserveSupported','maxStorageBufferRange')


def package_identity(bundle, library, require_phone):
    raw = (bundle/'manifest.json').read_bytes(); manifest = json.loads(raw)
    assert sha(raw) == CANDIDATE_MANIFEST_SHA, 'Only the frozen v3 package is admitted'
    # v3 repacks the unchanged constants into a smaller arena, so its complete
    # weights.bin has a different hash. Pin both layouts, never waive either.
    baseline = REPO/'tools/android-streaming-vulkan/generated/candidate-v2'
    old_raw = (baseline/'manifest.json').read_bytes(); assert sha(old_raw) == BASELINE_MANIFEST
    old = json.loads(old_raw)
    assert old['files']['weights.bin']['sha256'] == BASELINE_WEIGHTS_SHA
    assert old['files']['plan.bin']['sha256'] == BASELINE_PLAN_SHA
    for name,item in old['files'].items():
        value=(baseline/name).read_bytes()
        assert len(value) == integer(item['bytes']) and sha(value) == item['sha256'], f'Baseline integrity: {name}'
    assert manifest['layers'] == 8 and manifest['stateCount'] == 31 and manifest['precision'] == 'float32'
    assert manifest['selectedWeightBytesUnchanged'] is True
    assert manifest['files']['weights.bin']['sha256'] == CANDIDATE_WEIGHTS_SHA
    assert manifest['files']['plan.bin']['sha256'] == CANDIDATE_PLAN_SHA
    for name, item in manifest['files'].items():
        path = Path(name)
        assert not path.is_absolute() and '..' not in path.parts
        data = (bundle/path).read_bytes()
        assert len(data) == integer(item['bytes']) and sha(data) == item['sha256'], name
    plan = (bundle/'plan.bin').read_bytes(); assert plan[:8] == b'RVKPLAN1'
    version, floats, ops, inputs, outputs, states = struct.unpack_from('<6I', plan, 8)
    assert version == 1 and inputs == outputs == 33 and states == 31
    assert ops == 466 and floats*4 == 104000512
    assert len(plan) == 32+(inputs+outputs)*8+states*12+ops*132
    native = library.read_bytes()
    assert sha(native) == CANDIDATE_NATIVE_SHA, 'Only the frozen Android v3 native library is admitted'
    if require_phone: assert native[:4] == b'\x7fELF', 'Physical run needs the Android shared library'
    start = 32+(inputs+outputs)*8+states*12
    descriptors = [struct.unpack_from('<33I',plan,start+i*132) for i in range(ops)]
    return sha(raw), sha(native), ops, floats*4, descriptors


def gpu(vk, operations, arena, require_phone, profiled=False):
    assert vk['backend'] == 'native-cpp-vulkan-fp32'
    assert vk['hardwareGpu'] is True and vk['softwareDeviceAllowed'] is False
    assert vk['precision'] == 'float32' and integer(vk['stateCount']) == 31
    for key in ['statePingPong','gpuFiniteChecks','tiledMatmulPipeline','tiledConvolutionPipeline']:
        assert vk[key] is True, key
    assert vk['perDispatchProfiling'] is profiled
    assert integer(vk['operations']) == operations and integer(vk['arenaBytes']) == arena
    assert integer(vk['maxStorageBufferRange']) >= arena and integer(vk['timestampValidBits']) > 0
    for key in ['vendorId','deviceId','deviceType','driverVersion','apiVersion','subgroupSize','subgroupOperations']:
        assert integer(vk[key]) > 0
    if require_phone:
        for key, value in PHONE_IDENTITY.items(): assert vk[key] == value, key
    assert type(vk['float32DenormPreserveSupported']) is bool


def latency(case, label, frames, samples):
    assert case['caseId'] == label and case['backend'] == 'Vulkan' and integer(case['cpuThreads']) == 0
    assert integer(case['layers']) == 8 and case['precision'] == 'FP32'
    assert integer(case['frames']) == frames and case['allOutputFinite'] is True
    assert integer(case['inputSamples']) == integer(case['outputSamples']) == samples
    assert len(case['rawPushMs']) == len(case['rawRenderMs']) == frames
    times = np.asarray([number(x) for x in case['rawPushMs']], dtype=np.float64)
    renders = np.asarray([number(x) for x in case['rawRenderMs']], dtype=np.float64)
    assert (times > 0).all() and (renders > 0).all() and (renders <= times+0.01).all()
    number(case['checksum']); s = np.sort(times)
    calculated = {'mean':float(times.mean()),'min':float(s[0]),'max':float(s[-1])}
    calculated.update({key:float(s[math.ceil(q*frames)-1]) for key,q in [('p50',.5),('p95',.95),('p99',.99)]})
    for key,value in calculated.items(): assert abs(number(case['latencyMs'][key])-value) < 1e-6
    assert integer(case['over20Ms']) == int((times>20).sum())
    assert abs(number(case['rtf'])-times.mean()/20) < 1e-6
    return times


def quality(folder, run_id, label, reported, clean):
    waves = {}
    for name in ['input','cpu','vulkan']:
        raw = read_checked(folder/f'{run_id}-quality-{label}-{name}.f32')
        data = np.frombuffer(raw,dtype='<f4').astype(np.float64)
        assert len(data) == 64000 and np.isfinite(data).all()
        if name == 'input': assert sha(raw) == NOISY_SHA
        if name == 'cpu': assert sha(raw) == GOLDEN_CPU_SHA
        waves[name] = data
    a,b = waves['cpu'],waves['vulkan']; delta = b-a
    rms = float(np.sqrt(np.sum(delta*delta)/np.sum(a*a))); absolute = float(np.max(np.abs(delta)))
    assert absolute <= 5e-6 and rms <= 1e-5
    reference = reference_metrics(clean,a); candidate = reference_metrics(clean,b)
    stoi = candidate['stoi']-reference['stoi']; sisdr = candidate['siSdrDb']-reference['siSdrDb']
    assert abs(stoi) <= 1e-4 and abs(sisdr) <= .001
    assert reported['passed'] is True and integer(reported['samples']) == 64000
    assert abs(number(reported['maxAbs'])-absolute) < 1e-12
    assert abs(number(reported['relativeRms'])-rms) < 1e-12
    assert reported['resetBitwiseReplay'] is True and reported['partialTailsAndCancellation'] is True
    return {'maxAbs':absolute,'relativeRms':rms,'stoiDelta':stoi,'siSdrDeltaDb':sisdr,
            'scope':'One four-second synthetic diagnostic against pinned CPU output; no alignment, scaling or cropping'}


def expected_dispatch(words, vk):
    opcode = words[0]
    basic = {1:'unary',2:'binary',3:'where',4:'gather',5:'pad',6:'scatter',
             7:'matmul',8:'conv',9:'layernorm',10:'frequency',11:'temporal',13:'concat_cache'}
    assert opcode in basic, f'Unexpected opcode {opcode}'
    kernel = basic[opcode]
    groups = [(words[2]+127)//128,1,1]
    if opcode == 7:
        if vk['tiledMatmulPipeline']:
            kernel='matmul_tiled';groups=[(words[7]+31)//32,(words[6]+15)//16,words[9]]
        else:groups=[(words[7]+15)//16,(words[6]+7)//8,words[9]]
    elif opcode == 8:
        assert words[27] in (0,1)
        tiled = vk['tiledConvolutionPipeline'] and words[22] == 1 and words[24] == 0 and words[25] == 0 and words[26] == 0 and words[7] >= 16
        if words[27]:
            assert tiled and words[14] == 3 and words[15] == 3, 'Specialized3x3 flag requires admitted dense3x3'
        if tiled:
            kernel='conv_3x3_tiled' if words[27] else 'conv_tiled'
            groups=[(words[8]*words[9]+31)//32,(words[7]+15)//16,words[6]]
    elif opcode == 9:groups=[words[6],1,1]
    elif opcode == 10:groups=[(words[13]+3)//4,1,1]
    elif opcode == 11:groups=[(words[2]+3)//4,1,1]
    return kernel, groups


def optional_profile(value, vk, operations, arena, require_phone, descriptors):
    assert type(value['performed']) is bool
    if not value['performed']:
        assert isinstance(value['reason'],str) and value['reason']
        return
    assert value['profiledOutputBitwiseEqual'] is True and integer(value['frames']) == 4
    gpu(value['vulkan'],operations,arena,require_phone,True)
    for key in IDENTITY_KEYS: assert value['vulkan'][key] == vk[key]
    assert len(value['profiles']) == 4
    previous = None
    for p in value['profiles']:
        assert p['schemaVersion'] == 1 and p['profiled'] is True
        assert integer(p['operationCount']) == operations and integer(p['queryCount']) == operations*2+7
        ordinal = integer(p['frameOrdinal']); assert ordinal > 0
        if previous is not None: assert ordinal == previous+1
        previous = ordinal
        assert integer(p['timestampValidBits']) == vk['timestampValidBits']
        assert number(p['timestampPeriodNs']) > 0
        fields = ['wholeGpuMs','inputAndStatusInitMs','inputDependencyMs','computeGraphMs','summedDispatchMs',
                  'unattributedGraphMs','finiteValidationMs','outputReadbackMs','completionMs']
        for key in fields: assert number(p[key]) >= 0
        assert len(p['operations']) == operations
        categories = {}
        for i,op in enumerate(p['operations']):
            assert integer(op['index']) == i and integer(op['opcode']) == descriptors[i][0]
            assert integer(op['outputElements']) == descriptors[i][2]
            assert len(op['workgroups']) == 3 and all(integer(x)>0 for x in op['workgroups'])
            kernel,groups = expected_dispatch(descriptors[i],vk)
            assert op['kernel'] == kernel and op['workgroups'] == groups, f'Wrong pipeline/dispatch for operation {i}'
            duration = number(op['gpuMs']); assert duration >= 0
            totals = categories.setdefault(op['kernel'],[0,0.0]);totals[0]+=1;totals[1]+=duration
        assert abs(sum(x[1] for x in categories.values())-p['summedDispatchMs']) < 1e-6
        assert len(p['categories']) == len(categories)
        for c in p['categories']:
            n,total = categories.pop(c['kernel']);assert integer(c['dispatches']) == n
            assert abs(number(c['gpuMs'])-total) < 1e-6
        assert not categories
        assert abs(p['computeGraphMs']-p['summedDispatchMs']-p['unattributedGraphMs']) < 1e-6
        segments = ['inputAndStatusInitMs','inputDependencyMs','computeGraphMs','finiteValidationMs','outputReadbackMs','completionMs']
        assert abs(sum(p[k] for k in segments)-p['wholeGpuMs']) < 1e-6


def verify(folder, run_id, bundle, library, clean_file, require_phone=True, allow_structural_fixture=False):
    assert re.fullmatch(r'[A-Za-z0-9_.-]{1,80}',run_id)
    assert not (folder/f'{run_id}-watchdog.json').exists(), 'Process watchdog fired'
    expected_manifest,expected_native,operations,arena,descriptors = package_identity(bundle,library,require_phone)
    raw = read_checked(folder/f'{run_id}-result.json'); r = json.loads(raw)
    assert allow_structural_fixture or '_structuralFixtureOnly' not in r, 'Adapted fixtures are not benchmark evidence'
    assert integer(r['schemaVersion']) == 3 and r['runId'] == run_id and r['success'] is True
    assert integer(r['layers']) == 8 and r['precision'] == 'FP32'
    assert integer(r['sampleRate']) == 16000 and integer(r['hopSamples']) == 320
    assert integer(r['warmupHops']) == 16 and r['caseOrder'] == 'ABBA'
    assert integer(r['workDeadlineMs']) == 45000 and integer(r['processDeadlineMs']) == 50000
    assert 0 < integer(r['elapsedMs']) < 50000
    assert r['baselineBundleManifestSha256'] == BASELINE_MANIFEST
    assert r['bundleManifestSha256'] == expected_manifest and r['nativeLibrarySha256'] == expected_native
    d = r['device']; assert type(d['emulatorReported']) is bool
    if require_phone:
        assert d['emulatorReported'] is False and d['model'] == 'SM-S938U1' and integer(d['sdk']) == 35
        assert d['manufacturer'] == 'samsung' and d['abis'] == ['arm64-v8a'] and d['fingerprint'].startswith('samsung/')
    baseline,candidate = r['baselineVulkan'],r['vulkan']
    gpu(baseline,530,104001280,require_phone);gpu(candidate,operations,arena,require_phone)
    for key in IDENTITY_KEYS: assert baseline[key] == candidate[key], f'Device mismatch: {key}'
    frames = integer(r['timedFramesPerCase']); assert 32 <= frames <= 160
    if require_phone: assert frames == 160
    samples = 640+(16+frames)*320
    raw_input = read_checked(folder/f'{run_id}-benchmark-input.f32')
    assert len(raw_input) == samples*4 and np.isfinite(np.frombuffer(raw_input,dtype='<f4')).all()
    assert sha(raw_input) == r['benchmarkInputSha256']
    full = (REPO/'Validation/AndroidStreamingOptimization-vulkan/benchmark-input.f32').read_bytes()
    assert sha(full) == BENCHMARK_SHA and raw_input == full[:len(raw_input)]
    assert len(r['cases']) == 4
    labels = ['vulkan-v2-first','vulkan-v3-first','vulkan-v3-second','vulkan-v2-second']
    timings = [latency(c,label,frames,samples) for c,label in zip(r['cases'],labels)]
    assert sum(float(x.sum()) for x in timings) <= r['elapsedMs']+1, 'Measured time exceeds whole execution'
    assert r['cases'][0]['checksum'] == r['cases'][3]['checksum']
    assert r['cases'][1]['checksum'] == r['cases'][2]['checksum']
    clean_raw = clean_file.read_bytes();assert sha(clean_raw) == CLEAN_SHA
    clean = np.frombuffer(clean_raw,dtype='<f4').astype(np.float64)
    qualities = {label:quality(folder,run_id,label,r[field],clean)
                 for label,field in [('baseline','qualityBaseline'),('optimized','quality')]}
    optional_profile(r['optionalProfile'],candidate,operations,arena,require_phone,descriptors)
    base = np.concatenate([timings[0],timings[3]]);new = np.concatenate([timings[1],timings[2]])
    def summary(values):
        s=np.sort(values);return {'mean':float(values.mean()),'p50':float(s[math.ceil(.5*len(s))-1]),
          'p95':float(s[math.ceil(.95*len(s))-1]),'max':float(s[-1]),'over20':int((values>20).sum()),'count':len(values)}
    return {'verified':True,'resultSha256':sha(raw),'result':r,'independentQuality':qualities,
            'baseline':summary(base),'optimized':summary(new),'under20Mean':float(new.mean())<20,
            'under20EveryHop':float(new.max())<20,'latencyReductionFraction':float(1-new.mean()/base.mean()),
            'baselineFirstToLastDriftFraction':float(timings[3].mean()/timings[0].mean()-1),
            'candidateRepeatDriftFraction':float(timings[2].mean()/timings[1].mean()-1)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--candidate-bundle',type=Path,required=True);p.add_argument('--native-library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-emulator',action='store_true')
    p.add_argument('--clean',type=Path,default=REPO/'Validation/AndroidStreamingOptimization-0.1.2/quality/clean-speech.f32')
    a=p.parse_args();r=verify(a.folder,a.run_id,a.candidate_bundle,a.native_library,a.clean,not a.allow_emulator)
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='result'},indent=2))
if __name__=='__main__':main()
