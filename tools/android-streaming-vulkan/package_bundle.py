#!/usr/bin/env python3
"""Package a fixed, hash-checked FP32 Vulkan experiment; this does not certify numerical quality."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
BASELINE = '18db8d32b638702909de484d2a9cffc97ed09e6454c4caed3f0008fb0aab53e8'
SOURCE = 'db29c725c70608920e600787fcda625e98429cfbae2c2d09f1dc73726825319a'
CHECKPOINT = '6eac4e8cc721f99a22cde7eac3c9f8e2826ced5dc8a4234fd1977de1ec0d3ade'

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def pin(p): return {'bytes': p.stat().st_size, 'sha256': digest(p)}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generated', type=Path, default=ROOT/'generated')
    p.add_argument('--host-library', type=Path, required=True)
    p.add_argument('--android-library', type=Path, required=True)
    p.add_argument('--update-sdk-pins', action='store_true')
    a = p.parse_args()
    export = json.loads((a.generated/'export.json').read_text())
    assert export['layers'] == 8 and export['stateCount'] == 31 and export['precision'] == 'float32'
    assert export['checkpointSha256'] == CHECKPOINT and export['baselineModelSha256'] == BASELINE
    assert export['sourceModelSha256'] == SOURCE and export['selectedWeightBytesUnchanged'] is True
    assert export['recordWords'] == 33 and export['layoutFusion'] is True
    if export.get('concatCacheFusion', False):
        assert export['dispatchCount'] == 466 and export['fusedConcatCount'] == 41 and export['fusedCacheCount'] == 23
    else:
        assert export['dispatchCount'] == 530
    # Bind the recorded inventory to the actual serialized plan, not just JSON.
    plan = (a.generated/'plan.bin').read_bytes()
    assert plan[:8] == b'RVKPLAN1'
    version, arena_floats, operations, inputs, outputs, states = struct.unpack_from('<6I', plan, 8)
    assert version == 1 and operations == export['dispatchCount'] and inputs == outputs == 33 and states == 31
    assert arena_floats * 4 == export['arenaBytes']
    op_start = 32 + (inputs + outputs) * 8 + states * 12
    assert len(plan) == op_start + operations * 132
    concat_records = [struct.unpack_from('<33I', plan, op_start + i * 132)
                      for i in range(operations) if struct.unpack_from('<I', plan, op_start + i * 132)[0] == 13]
    if export.get('concatCacheFusion', False):
        assert len(concat_records) == 41 and sum(record[10] for record in concat_records) == 23
    else:
        assert not concat_records
    specialized_records = [struct.unpack_from('<33I', plan, op_start + i * 132)
                           for i in range(operations)
                           if struct.unpack_from('<I', plan, op_start + i * 132)[0] == 8 and
                           struct.unpack_from('<I', plan, op_start + i * 132 + 27 * 4)[0] == 1]
    if export.get('specializedConv3x3', False):
        assert export['specializedConv3x3Count'] == len(specialized_records) == 13
    else:
        assert not specialized_records
    shaders = a.generated/'shaders'
    proof = json.loads((shaders/'shader-receipt.json').read_text())
    assert proof['float16'] is False and proof['relaxedPrecision'] is False
    assert len(proof['kernels']) >= 12
    if specialized_records:
        assert 'conv_3x3_tiled' in proof['kernels']
    files = {name: pin(a.generated/name) for name in ['plan.bin', 'weights.bin']}
    for name, record in files.items():
        assert record == export['files'][name], f'Export receipt does not match {name}'
    for name, evidence in proof['kernels'].items():
        assert re.fullmatch('[a-z0-9_]+', name)
        source = shaders/(name+'.spv')
        assert evidence['floatTypes'] == [32] and evidence['spirvSha256'] == digest(source)
        files['shaders/'+source.name] = pin(source)
    manifest = {
        'schemaVersion': 1, 'executionBackend': 'native-vulkan-fp32', 'precision': 'float32',
        'layers': 8, 'sampleRate': 16000, 'hopSize': 320, 'fftSize': 640, 'stateCount': 31,
        'checkpointSha256': CHECKPOINT, 'baselineModelSha256': BASELINE, 'sourceModelSha256': SOURCE,
        'selectedWeightBytesUnchanged': True, 'license': 'NSCLv1', 'researchOnly': True,
        'exportReceiptSha256': digest(a.generated/'export.json'),
        'shaderReceiptSha256': digest(shaders/'shader-receipt.json'), 'files': files,
    }
    (a.generated/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    assert a.android_library.name == 'libreuse_streaming_vulkan.so'
    assert a.host_library.name == 'libreuse_streaming_vulkan.dylib'
    receipt = {'bundleManifest': pin(a.generated/'manifest.json'),
               'hostLibrary': pin(a.host_library), 'androidLibrary': pin(a.android_library)}
    (a.generated/'package-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    if a.update_sdk_pins:
        sdk = REPO/'android/reuse-streaming/src/main/kotlin/io/riftco/reusestreaming/ExperimentalVulkanStreaming.kt'
        text = sdk.read_text()
        for key, value in [('BUNDLE_MANIFEST_SHA256', digest(a.generated/'manifest.json')),
                           ('ANDROID_LIBRARY_SHA256', digest(a.android_library)),
                           ('HOST_LIBRARY_SHA256', digest(a.host_library))]:
            text, count = re.subn(r'(internal const val '+key+r' = ")[^"]+("\n)', lambda m: m[1]+value+m[2], text)
            assert count == 1
        sdk.write_text(text)
    assets = REPO/'android/benchmarks/streaming/app/build/benchmark-vulkan-assets/vulkan'
    assets.mkdir(parents=True, exist_ok=True)
    for name in ['manifest.json']+list(files):
        dest = assets/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(a.generated/name, dest)
    libs = REPO/'android/benchmarks/streaming/app/build/benchmark-vulkan-libs/arm64-v8a'
    libs.mkdir(parents=True, exist_ok=True)
    shutil.copy2(a.android_library, libs/a.android_library.name)
    # Pin the exact same three-tone/noise sequence used in the earlier physical CPU benchmark.
    pcm = bytearray(); noise = 0x1234ABCD
    for i in range(640+(32+300)*320):
        noise=(noise^(noise<<13))&0xffffffff; noise=(noise^(noise>>17))&0xffffffff; noise=(noise^(noise<<5))&0xffffffff
        random=(noise>>8)/8388607.5-1
        value=.12*math.sin(2*math.pi*173*i/16000)+.07*math.sin(2*math.pi*317*i/16000)+.025*math.sin(2*math.pi*911*i/16000)+.025*random
        pcm.extend(struct.pack('<f',value))
    assert hashlib.sha256(pcm).hexdigest() == 'e3150c5d89ff157b323627de4e5ede75a35be2e44088b284daa9d47268ccb9ca'
    benchmark = assets.parent/'validation/vulkan-benchmark.f32'
    benchmark.parent.mkdir(parents=True,exist_ok=True);benchmark.write_bytes(pcm)
    print(json.dumps(receipt, indent=2))

if __name__ == '__main__': main()
