#!/usr/bin/env python3
"""Compare native Vulkan model variants on one local GPU, sequentially.

This is a network-only diagnostic, not the public streaming push timer and not
an Android-phone result. Every case uses the same input sequence and excludes
the first eight invocations. Baseline controls bracket the candidate runs.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli',type=Path,default=ROOT/'generated/build-host/reuse-vulkan-cli')
    parser.add_argument('--plan',type=Path,default=ROOT/'generated/plan.bin')
    parser.add_argument('--weights',type=Path,default=ROOT/'generated/weights.bin')
    parser.add_argument('--input',type=Path,default=ROOT/'generated/oracle/input.f32')
    parser.add_argument('--output',type=Path,default=ROOT/'generated/shader-comparison')
    parser.add_argument('--frames',type=int,default=64)
    parser.add_argument('--strict-shaders',default='shaders-baseline')
    parser.add_argument('--fma-shaders',default='shaders-fma')
    parser.add_argument('--cases',help='Comma-separated subset of the named comparison cases')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    source=np.fromfile(args.input,dtype='<f4').reshape(-1,642)
    data=np.tile(source,(int(np.ceil(args.frames/len(source))),1))[:args.frames]
    data.tofile(args.output/'input.f32')
    cases=[('strict-before','shaders-baseline',[]),('fma','shaders-fma',[]),
           ('fma-matmul','shaders-fma',['--tiled-matmul']),('fma-conv','shaders-fma',['--tiled-conv']),
           ('fma-both','shaders-fma',['--tiled-matmul','--tiled-conv']),
           ('strict-matmul','shaders-baseline',['--tiled-matmul']),
           ('strict-conv','shaders-baseline',['--tiled-conv']),
           ('strict-both','shaders-baseline',['--tiled-matmul','--tiled-conv']),
           ('strict-after','shaders-baseline',[])]
    if args.cases:
        selected=args.cases.split(',')
        known={v[0] for v in cases}
        if any(v not in known for v in selected):raise ValueError('Unknown comparison case')
        cases=[next(v for v in cases if v[0]==name) for name in selected]
    entries=[];baseline=None
    for name,shaders,flags in cases:
        shaders=args.strict_shaders if shaders=='shaders-baseline' else args.fma_shaders
        directory=args.output/name;directory.mkdir(exist_ok=True)
        command=[str(args.cli),str(args.plan),str(args.weights),str(ROOT/'generated'/shaders),
                 str(args.output/'input.f32'),str(directory/'output.f32'),'--states',str(directory/'states')]+flags
        result=subprocess.run(command,text=True,capture_output=True,timeout=180)
        (directory/'stdout.txt').write_text(result.stdout);(directory/'stderr.txt').write_text(result.stderr)
        if result.returncode:raise RuntimeError(f'{name}: {result.stderr[-4000:]}')
        receipt=json.loads(result.stdout);times=np.asarray(receipt['frameMs'][8:]);gpu=np.asarray(receipt['gpuMs'][8:])
        output=np.fromfile(directory/'output.f32',dtype='<f4')
        if baseline is None:baseline=output
        entry={'case':name,'frames':args.frames,'excludedWarmup':8,'meanMs':float(times.mean()),
               'medianMs':float(np.median(times)),'p95Ms':float(np.percentile(times,95)),
               'gpuMedianMs':float(np.median(gpu)),'maxOutputAbsVsStrict':float(np.max(abs(output-baseline))),
               'device':receipt['device'],'flags':flags,'command':command}
        entries.append(entry);print(json.dumps({k:v for k,v in entry.items() if k not in ['device','command']}),flush=True)
    (args.output/'summary.json').write_text(json.dumps({'scope':'local GPU network-only','cases':entries},indent=2)+'\n')

if __name__=='__main__':main()
