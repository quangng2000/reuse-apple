#!/usr/bin/env python3
"""Negative tests derived from a completed, independently valid device result.

Run only after an emulator smoke or physical result has been collected. This
suite never launches a cloud test; each mutation is confined to a temporary
copy of the exported evidence.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

import verify_phone_result as verifier


class ResultValidationTests(unittest.TestCase):
    arguments=None

    @classmethod
    def setUpClass(cls):
        args=cls.arguments
        cls.source=Path(args.folder)
        cls.run_id=args.run_id
        cls.manifest=args.expected_manifest
        cls.clean=Path(args.clean)
        cls.phone=not args.allow_emulator
        # Every negative case starts from evidence that first passes the real
        # independent statistics, waveform and quality checks.
        cls.valid=verifier.verify(cls.source,cls.run_id,cls.manifest,cls.clean,cls.phone)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='reuse-vulkan-result-negative-')
        self.folder=Path(self.temp.name)
        for source in self.source.glob(self.run_id+'-*'):
            if source.is_file():shutil.copy2(source,self.folder/source.name)
        self.result=copy.deepcopy(self.valid['result'])

    def tearDown(self):self.temp.cleanup()

    def save_result(self):
        data=json.dumps(self.result,indent=2,allow_nan=True).encode()
        target=self.folder/(self.run_id+'-result.json')
        target.write_bytes(data)
        target.with_suffix('.sha256').write_text(verifier.sha(data)+'\n')

    def save_audio(self,label,data):
        path=self.folder/(self.run_id+'-'+label+'.f32')
        path.write_bytes(data);Path(str(path)+'.sha256').write_text(verifier.sha(data)+'\n')

    def rejected(self,physical=None):
        self.save_result()
        with self.assertRaises((AssertionError,ValueError,KeyError,TypeError)):
            verifier.verify(self.folder,self.run_id,self.manifest,self.clean,self.phone if physical is None else physical)

    def test_manifest_mismatch(self):
        self.result['bundleManifestSha256']='0'*64;self.rejected()

    def test_wrong_number_of_layers(self):
        self.result['layers']=4;self.rejected()

    def test_wrong_warmup_policy(self):
        self.result['warmupHops']=0;self.rejected()

    def test_reduced_precision(self):
        self.result['precision']='FP16';self.rejected()

    def test_wrong_arena(self):
        self.result['vulkan']['arenaBytes']+=256;self.rejected()

    def test_tiled_wrong_arena(self):
        self.result['vulkanTiled']['arenaBytes']+=256;self.rejected()

    def test_wrong_graph_operation_count(self):
        self.result['vulkan']['operations']=750;self.rejected()

    def test_unrequested_cpu_fallback(self):
        self.result['vulkan']['hardwareGpu']=False;self.rejected()

    def test_allowed_software_driver(self):
        self.result['vulkan']['softwareDeviceAllowed']=True;self.rejected()

    def test_missing_persistent_states(self):
        self.result['vulkan']['stateCount']=30;self.rejected()

    def test_wrong_tiled_kernel_selection(self):
        self.result['vulkanTiled']['tiledMatmulPipeline']=False;self.rejected()

    def test_physical_run_requires_300_hops(self):
        # Force the expected identity only in this rejected metadata fixture
        # so an emulator-derived test still reaches the physical length gate.
        self.result['device'].update(model='SM-S938U1',sdk=35,emulatorReported=False)
        self.result['vulkan']['deviceName']='Adreno test mutation'
        self.result['timedFramesPerCase']=299
        self.rejected(physical=True)

    def test_sidecar_corruption(self):
        self.save_result()
        (self.folder/(self.run_id+'-result.sha256')).write_text('0'*64+'\n')
        with self.assertRaises(AssertionError):
            verifier.verify(self.folder,self.run_id,self.manifest,self.clean,self.phone)

    def test_forged_mean(self):
        self.result['cases'][1]['latencyMs']['mean']-=1;self.rejected()

    def test_forged_percentile(self):
        self.result['cases'][2]['latencyMs']['p95']-=1;self.rejected()

    def test_missing_timing_sample(self):
        self.result['cases'][1]['rawPushMs'].pop();self.rejected()

    def test_nonfinite_timing(self):
        self.result['cases'][1]['rawPushMs'][0]=float('nan');self.rejected()

    def test_wrong_output_sample_count(self):
        self.result['cases'][1]['outputSamples']-=1;self.rejected()

    def test_benchmark_input_identity_even_with_new_sidecar(self):
        data=bytearray((self.folder/(self.run_id+'-benchmark-input.f32')).read_bytes())
        data[0]^=1
        self.save_audio('benchmark-input',bytes(data))
        self.result['benchmarkInputSha256']=verifier.sha(data)
        self.rejected()

    def test_changed_waveform_even_with_new_sidecar(self):
        data=np.fromfile(self.folder/(self.run_id+'-quality-tiled-vulkan.f32'),dtype='<f4')
        data[100]+=.001
        self.save_audio('quality-tiled-vulkan',data.tobytes());self.rejected()

    def test_nonfinite_waveform(self):
        data=np.fromfile(self.folder/(self.run_id+'-quality-direct-vulkan.f32'),dtype='<f4')
        data[100]=np.nan
        self.save_audio('quality-direct-vulkan',data.tobytes());self.rejected()

    def test_forged_quality_error(self):
        self.result['quality']['maxAbs']+=1e-6;self.rejected()

    def test_reset_flag_not_boolean(self):
        self.result['quality']['resetBitwiseReplay']='true';self.rejected()

    def test_twenty_ms_exact_is_not_under_twenty(self):
        for case in self.result['cases'][1:3]:
            count=case['frames']
            case['rawPushMs']=[20.0]*count
            case['rawRenderMs']=[19.0]*count
            case['latencyMs']={k:20.0 for k in ['mean','min','max','p50','p95','p99']}
            case['over20Ms']=0;case['rtf']=1.0
        self.save_result()
        checked=verifier.verify(self.folder,self.run_id,self.manifest,self.clean,self.phone)
        self.assertFalse(checked['under20Mean'])
        self.assertFalse(checked['under20EveryHop'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,required=True);parser.add_argument('--run-id',required=True)
    parser.add_argument('--expected-manifest',required=True)
    parser.add_argument('--clean',type=Path,default=verifier.REPO/'Validation/AndroidStreamingOptimization-0.1.2/quality/clean-speech.f32')
    parser.add_argument('--allow-emulator',action='store_true')
    args=parser.parse_args();ResultValidationTests.arguments=args
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ResultValidationTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=='__main__':main()
