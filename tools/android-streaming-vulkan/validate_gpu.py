"""Validate every GPU-carried state and spectra against pinned FP32 ORT.

Host Vulkan execution is correctness evidence, not a physical-phone benchmark.
The recorded tolerances concern internal features; the separate public SDK
waveform/audio quality gate remains stricter and must pass before publication.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import onnxruntime as ort

from export_plan import HERE,SOURCE,SOURCE_SHA,digest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,default=HERE/'generated')
    parser.add_argument('--shaders',type=Path)
    parser.add_argument('--cli',type=Path,default=HERE/'generated/build-host/reuse-vulkan-cli')
    parser.add_argument('--output',type=Path,default=HERE/'generated/gpu-validation')
    parser.add_argument('--frames',type=int,default=8)
    parser.add_argument('--tiled-conv',action='store_true')
    parser.add_argument('--tiled-matmul',action='store_true')
    args=parser.parse_args()
    assert 4<=args.frames<=128
    assert ort.__version__=='1.23.2' and digest(SOURCE)==SOURCE_SHA
    args.output.mkdir(parents=True,exist_ok=True)
    shaders=args.shaders or args.directory/'shaders'
    options=ort.SessionOptions();options.log_severity_level=3;options.intra_op_num_threads=1
    library=SOURCE.parent/'build-host/libreuse_streaming_stable_v2.dylib'
    options.register_custom_ops_library(str(library))
    reference=ort.InferenceSession(str(SOURCE),options,providers=['CPUExecutionProvider'])
    states={v.name:np.zeros(v.shape,dtype=np.float32) for v in reference.get_inputs()[2:]}
    rng=np.random.default_rng(728);input_frames=[];expected=[]
    for frame in range(args.frames):
        magnitude=np.abs(rng.normal(0,.05,(1,321,1))).astype(np.float32)
        phase=rng.uniform(-3,3,(1,321,1)).astype(np.float32)
        if frame==0:magnitude[:]=0;phase[:]=0
        if frame==1:magnitude[:]=0;magnitude[0,100,0]=1;phase[:]=0
        input_frames.append(np.concatenate([magnitude.reshape(-1),phase.reshape(-1)]))
        output=reference.run(None,{'magnitude':magnitude,'phase':phase,**states})
        expected.append(np.concatenate([output[0].reshape(-1),output[1].reshape(-1)]))
        for info,value in zip(reference.get_inputs()[2:],output[2:]):states[info.name]=value
    np.array(input_frames,dtype='<f4').tofile(args.output/'input.f32')
    np.array(expected,dtype='<f4').tofile(args.output/'expected.f32')
    env=os.environ.copy()
    if 'VK_ICD_FILENAMES' not in env:
        path=Path('/opt/homebrew/opt/molten-vk/etc/vulkan/icd.d/MoltenVK_icd.json')
        if path.exists():env['VK_ICD_FILENAMES']=str(path)
    command=[str(args.cli),str(args.directory/'plan.bin'),str(args.directory/'weights.bin'),str(shaders),str(args.output/'input.f32'),str(args.output/'actual.f32'),'--states',str(args.output/'states'),'--reset-replay']
    if args.tiled_conv:command.append('--tiled-conv')
    if args.tiled_matmul:command.append('--tiled-matmul')
    result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=180,check=True)
    (args.output/'runtime.stdout.json').write_text(result.stdout)
    (args.output/'runtime.stderr.txt').write_text(result.stderr)
    runtime=json.loads(result.stdout)
    assert runtime['device']['hardwareGpu'] and runtime['resetReplayBitwise']
    actual=np.fromfile(args.output/'actual.f32',dtype='<f4').reshape(args.frames,642)
    expected=np.array(expected,dtype=np.float32)
    errors=[]
    def metric(name,ref,got):
        assert ref.shape==got.shape and np.isfinite(got).all(),name
        delta=got.astype(np.float64)-ref
        rms=float(np.linalg.norm(delta)/max(float(np.linalg.norm(ref)),1e-30))
        return {'name':name,'maxAbs':float(np.max(np.abs(delta))),'relativeRms':rms}
    for i in range(args.frames):
        errors.append(metric(f'frame{i}:magnitude',expected[i,:321],actual[i,:321]))
        errors.append(metric(f'frame{i}:phase',expected[i,321:],actual[i,321:]))
    state_errors=[]
    for i,(name,ref) in enumerate(states.items()):
        got=np.fromfile(args.output/f'states/state-{i}.f32',dtype='<f4').reshape(ref.shape)
        state_errors.append(metric(name,ref,got))
    # Spectral phase can change rapidly near zero complex components, so
    # report complex-spectrum agreement as well as raw radians.
    re=expected[:,:321]*np.cos(expected[:,321:]);im=expected[:,:321]*np.sin(expected[:,321:])
    ar=actual[:,:321]*np.cos(actual[:,321:]);ai=actual[:,:321]*np.sin(actual[:,321:])
    complex_error=metric('complexSpectrum',np.stack([re,im]),np.stack([ar,ai]))
    policy={'spectrumComplexMaxAbs':5e-6,'spectrumComplexRelativeRms':1e-5,'stateMaxAbs':1e-4,'stateRelativeRms':1e-5}
    passed=complex_error['maxAbs']<=policy['spectrumComplexMaxAbs'] and complex_error['relativeRms']<=policy['spectrumComplexRelativeRms'] and all(v['maxAbs']<=policy['stateMaxAbs'] and v['relativeRms']<=policy['stateRelativeRms'] for v in state_errors)
    receipt={'schemaVersion':1,'scope':'host hardware GPU numerical correctness; not phone latency','frames':args.frames,'sourceModelSha256':SOURCE_SHA,'planSha256':digest(args.directory/'plan.bin'),'weightsSha256':digest(args.directory/'weights.bin'),'shaders':{p.name:digest(p) for p in sorted(shaders.glob('*.spv'))},'inputSha256':digest(args.output/'input.f32'),'runtime':runtime,'policy':policy,'spectra':errors,'complexSpectrum':complex_error,'states':state_errors,'passed':passed}
    (args.output/'validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'passed':passed,'frames':args.frames,'complexSpectrum':complex_error,'maximumStateAbsoluteError':max(x['maxAbs'] for x in state_errors),'maximumStateRelativeError':max(x['relativeRms'] for x in state_errors),'device':runtime['device']['deviceName']},indent=2))
    assert passed,'GPU numerical gate failed'


if __name__=='__main__':main()
