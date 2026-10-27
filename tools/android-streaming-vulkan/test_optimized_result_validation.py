#!/usr/bin/env python3
"""Strict schema-2 verifier regressions using historical data as a STRUCTURAL FIXTURE.

No new phone results are fabricated or saved as benchmark evidence. The old
physical run is independently verified first, then copied into temporary
schema-2-shaped test data with an explicit fixture marker. Production verifier
calls reject that marker. No test makes a cloud call.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

import verify_optimized_result as verifier
import verify_phone_result as historical

DEFAULT_FOLDER = verifier.REPO/'Validation/AndroidStreamingOptimization-vulkan/physical-download/pa3q-35-en-portrait/artifacts/sdcard/Android/data/io.riftco.reusestreaming.benchmark/files/firebase-vulkan'
DEFAULT_RUN = 'reuse-vulkan-fp32-20261026-final'
CLEAN = verifier.REPO/'Validation/AndroidStreamingOptimization-0.1.2/quality/clean-speech.f32'


class OptimizedResultValidationTests(unittest.TestCase):
    arguments = None

    @classmethod
    def setUpClass(cls):
        args = cls.arguments
        folder = args.historical_folder if args else DEFAULT_FOLDER
        old_run = args.historical_run_id if args else DEFAULT_RUN
        old = historical.verify(folder, old_run, verifier.BASELINE_MANIFEST, CLEAN, True)['result']
        cls.fixture = tempfile.TemporaryDirectory(prefix='reuse-STRUCTURAL-NOT-BENCHMARK-')
        cls.source = Path(cls.fixture.name)
        cls.run_id = 'structural-fixture-not-a-new-benchmark'
        cls.manifest = 'f'*64  # Deliberately a fixture identity, not a claimed release.
        cls.valid = copy.deepcopy(old)
        cls.valid.update(schemaVersion=2, runId=cls.run_id,
                         baselineBundleManifestSha256=verifier.BASELINE_MANIFEST,
                         bundleManifestSha256=cls.manifest, elapsedMs=80000,
                         _structuralFixtureOnly='Adapted old results for validator tests; no new run')
        cls.valid['baselineVulkan'] = copy.deepcopy(old['vulkanTiled'])
        cls.valid['vulkan'] = copy.deepcopy(old['vulkanTiled'])
        for key in ['baselineVulkan','vulkan']: cls.valid[key]['perDispatchProfiling'] = False
        cls.valid.pop('vulkanTiled'); cls.valid.pop('qualityTiled')
        cls.valid['quality'] = copy.deepcopy(old['qualityTiled'])
        # Both GPU cases intentionally contain the same historical tiled run;
        # no candidate performance claim is inferred from this fixture.
        cls.valid['cases'][1] = copy.deepcopy(old['cases'][2])
        cls.valid['cases'][2] = copy.deepcopy(old['cases'][2])
        cls.valid['cases'][1]['caseId'] = 'vulkan-tiled-baseline'
        cls.valid['cases'][2]['caseId'] = 'vulkan-tiled-optimized'
        for old_label,new_label in [('benchmark-input','benchmark-input')]+[
                ('quality-tiled-'+name,'quality-optimized-'+name) for name in ['input','cpu','vulkan']]:
            src = folder/(old_run+'-'+old_label+'.f32')
            dst = cls.source/(cls.run_id+'-'+new_label+'.f32')
            shutil.copy2(src,dst); shutil.copy2(Path(str(src)+'.sha256'),Path(str(dst)+'.sha256'))
        raw = json.dumps(cls.valid,indent=2).encode()
        (cls.source/(cls.run_id+'-result.json')).write_bytes(raw)
        (cls.source/(cls.run_id+'-result.sha256')).write_text(verifier.sha(raw)+'\n')
        verifier.verify(cls.source,cls.run_id,cls.manifest,CLEAN,True,allow_structural_fixture=True)

    @classmethod
    def tearDownClass(cls): cls.fixture.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='reuse-v2-verifier-negative-')
        self.folder = Path(self.temp.name)
        for source in self.source.iterdir(): shutil.copy2(source,self.folder/source.name)
        self.result = copy.deepcopy(self.valid)

    def tearDown(self): self.temp.cleanup()

    def save(self):
        data = json.dumps(self.result,indent=2,allow_nan=True).encode()
        (self.folder/(self.run_id+'-result.json')).write_bytes(data)
        (self.folder/(self.run_id+'-result.sha256')).write_text(verifier.sha(data)+'\n')

    def save_audio(self,label,data):
        p = self.folder/(self.run_id+'-'+label+'.f32')
        p.write_bytes(data); Path(str(p)+'.sha256').write_text(verifier.sha(data)+'\n')

    def verify(self):
        return verifier.verify(self.folder,self.run_id,self.manifest,CLEAN,True,allow_structural_fixture=True)

    def rejected(self):
        self.save()
        with self.assertRaises((AssertionError,ValueError,KeyError,TypeError)): self.verify()

    def test_production_verifier_rejects_adapted_fixture(self):
        with self.assertRaisesRegex(AssertionError,'fixtures'):
            verifier.verify(self.folder,self.run_id,self.manifest,CLEAN,True)

    def test_fixture_is_self_consistent_without_claiming_new_measurements(self):
        v = self.verify(); self.assertEqual(v['gpuSpeedup'],1.0)
        self.assertTrue(v['result']['_structuralFixtureOnly'])

    def test_schema_mismatch(self): self.result['schemaVersion']=1; self.rejected()
    def test_wrong_candidate_manifest(self): self.result['bundleManifestSha256']='0'*64; self.rejected()
    def test_wrong_baseline_manifest(self): self.result['baselineBundleManifestSha256']='0'*64; self.rejected()
    def test_wrong_layers(self): self.result['layers']=4; self.rejected()
    def test_wrong_precision(self): self.result['precision']='FP16'; self.rejected()
    def test_wrong_warmup(self): self.result['warmupHops']=0; self.rejected()
    def test_unbounded_elapsed(self): self.result['elapsedMs']=94000; self.rejected()
    def test_wrong_graph(self): self.result['vulkan']['operations']=632; self.rejected()
    def test_wrong_arena(self): self.result['vulkan']['arenaBytes']+=256; self.rejected()
    def test_wrong_baseline_arena(self): self.result['baselineVulkan']['arenaBytes']+=256; self.rejected()
    def test_wrong_states(self): self.result['vulkan']['stateCount']=30; self.rejected()
    def test_missing_pingpong(self): self.result['vulkan']['statePingPong']=False; self.rejected()
    def test_no_finite_checks(self): self.result['vulkan']['gpuFiniteChecks']=False; self.rejected()
    def test_cpu_fallback(self): self.result['vulkan']['hardwareGpu']=False; self.rejected()
    def test_software_device(self): self.result['vulkan']['softwareDeviceAllowed']=True; self.rejected()
    def test_profiled_candidate_timings(self): self.result['vulkan']['perDispatchProfiling']=True; self.rejected()
    def test_profiled_baseline_timings(self): self.result['baselineVulkan']['perDispatchProfiling']=True; self.rejected()
    def test_missing_profile_flag(self): del self.result['vulkan']['perDispatchProfiling']; self.rejected()
    def test_direct_pipeline(self): self.result['vulkan']['tiledMatmulPipeline']=False; self.rejected()
    def test_wrong_driver(self): self.result['vulkan']['driverVersion']+=1; self.rejected()
    def test_wrong_gpu(self): self.result['vulkan']['deviceId']+=1; self.rejected()
    def test_wrong_subgroup(self): self.result['vulkan']['subgroupSize']=32; self.rejected()
    def test_emulator(self): self.result['device']['emulatorReported']=True; self.rejected()
    def test_wrong_phone(self): self.result['device']['model']='SM-OTHER'; self.rejected()
    def test_short_physical_run(self): self.result['timedFramesPerCase']=299; self.rejected()
    def test_wrong_case_order(self): self.result['cases'][1],self.result['cases'][2]=self.result['cases'][2],self.result['cases'][1]; self.rejected()
    def test_wrong_sample_count(self): self.result['cases'][2]['outputSamples']-=1; self.rejected()
    def test_forged_mean(self): self.result['cases'][2]['latencyMs']['mean']-=1; self.rejected()
    def test_forged_p95(self): self.result['cases'][2]['latencyMs']['p95']-=1; self.rejected()
    def test_missing_timing(self): self.result['cases'][2]['rawPushMs'].pop(); self.rejected()
    def test_nonfinite_timing(self): self.result['cases'][2]['rawPushMs'][0]=float('nan'); self.rejected()
    def test_boolean_timing(self): self.result['cases'][2]['rawPushMs'][0]=True; self.rejected()
    def test_forged_over20(self): self.result['cases'][2]['over20Ms']-=1; self.rejected()
    def test_forged_rtf(self): self.result['cases'][2]['rtf']-=.1; self.rejected()
    def test_reset_flag_not_boolean(self): self.result['quality']['resetBitwiseReplay']='true'; self.rejected()
    def test_forged_quality_error(self): self.result['quality']['maxAbs']+=1e-6; self.rejected()

    def test_sidecar_integrity(self):
        (self.folder/(self.run_id+'-result.sha256')).write_text('0'*64+'\n')
        with self.assertRaises(AssertionError): self.verify()

    def test_modified_benchmark_with_matching_new_sidecar(self):
        p=self.folder/(self.run_id+'-benchmark-input.f32'); data=bytearray(p.read_bytes()); data[0]^=1
        self.save_audio('benchmark-input',data); self.result['benchmarkInputSha256']=verifier.sha(data); self.rejected()

    def test_modified_golden_reference_with_matching_new_sidecar(self):
        p=self.folder/(self.run_id+'-quality-optimized-cpu.f32'); data=bytearray(p.read_bytes()); data[0]^=1
        self.save_audio('quality-optimized-cpu',data); self.rejected()

    def test_waveform_gate(self):
        data=np.fromfile(self.folder/(self.run_id+'-quality-optimized-vulkan.f32'),dtype='<f4'); data[100]+=.001
        self.save_audio('quality-optimized-vulkan',data.tobytes()); self.rejected()

    def test_nonfinite_waveform(self):
        data=np.fromfile(self.folder/(self.run_id+'-quality-optimized-vulkan.f32'),dtype='<f4'); data[100]=np.nan
        self.save_audio('quality-optimized-vulkan',data.tobytes()); self.rejected()

    def test_exact_twenty_is_not_under_twenty(self):
        case=self.result['cases'][2]; frames=case['frames']
        case.update(rawPushMs=[20.0]*frames,rawRenderMs=[19.0]*frames,
                    latencyMs={k:20.0 for k in ['mean','min','max','p50','p95','p99']},over20Ms=0,rtf=1.0)
        self.save(); v=self.verify()
        self.assertFalse(v['under20Mean']); self.assertFalse(v['under20EveryHop'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--historical-folder',type=Path,default=DEFAULT_FOLDER)
    p.add_argument('--historical-run-id',default=DEFAULT_RUN)
    OptimizedResultValidationTests.arguments=p.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(OptimizedResultValidationTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=='__main__': main()
