#!/usr/bin/env python3
"""Negative schema-3 verifier tests using explicitly synthetic/adapted fixtures."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import verify_final_budget_result as target

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO/'Validation/AndroidStreamingOptimization-vulkan-v2/comparison-download'
CANDIDATE = REPO/'tools/android-streaming-vulkan/generated/candidate-v3'
LIBRARY = CANDIDATE/'build-arm64-v8a/libreuse_streaming_vulkan.so'
CLEAN = REPO/'Validation/AndroidStreamingOptimization-0.1.2/quality/clean-speech.f32'


class FinalBudgetVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_result = next(SOURCE.rglob('*-result.json'))
        cls.original = json.loads(cls.source_result.read_text())
        cls.source_folder = cls.source_result.parent

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name); self.bundle = self.folder/'bundle';self.bundle.mkdir()
        manifest = json.loads((CANDIDATE/'manifest.json').read_text())
        for name in manifest['files']:
            p=self.bundle/name;p.parent.mkdir(parents=True,exist_ok=True);p.symlink_to(CANDIDATE/name)
        (self.bundle/'manifest.json').write_bytes((CANDIDATE/'manifest.json').read_bytes())
        self.run_id='schema3-structural-fixture'
        self.result=copy.deepcopy(self.original)
        r=self.result;r['_structuralFixtureOnly']=True;r['schemaVersion']=3;r['runId']=self.run_id
        r['baselineBundleManifestSha256']=target.BASELINE_MANIFEST
        r['bundleManifestSha256']=target.sha((self.bundle/'manifest.json').read_bytes())
        r['nativeLibrarySha256']=target.sha(LIBRARY.read_bytes())
        r['vulkan']['operations']=466;r['vulkan']['arenaBytes']=104000512
        r['caseOrder']='ABBA';r['warmupHops']=16;r['timedFramesPerCase']=160
        r['workDeadlineMs']=45000;r['processDeadlineMs']=50000;r['elapsedMs']=35000
        r['qualityBaseline']=copy.deepcopy(r['quality'])
        r['optionalProfile']={'performed':False,'reason':'fixture intentionally has no profile'}
        original=self.original['cases']; r['cases']=[copy.deepcopy(original[i]) for i in [1,2,2,1]]
        labels=['vulkan-v2-first','vulkan-v3-first','vulkan-v3-second','vulkan-v2-second']
        for c,label in zip(r['cases'],labels):
            c['caseId']=label;c['frames']=160;c['rawPushMs']=c['rawPushMs'][:160];c['rawRenderMs']=c['rawRenderMs'][:160]
            c['inputSamples']=c['outputSamples']=640+176*320
            times=np.asarray(c['rawPushMs']);s=np.sort(times)
            c['latencyMs']={'mean':float(times.mean()),'min':float(s[0]),'max':float(s[-1])}
            c['latencyMs'].update({k:float(s[math.ceil(q*160)-1]) for k,q in [('p50',.5),('p95',.95),('p99',.99)]})
            c['over20Ms']=int((times>20).sum());c['rtf']=float(times.mean()/20)
        full=(REPO/'Validation/AndroidStreamingOptimization-vulkan/benchmark-input.f32').read_bytes()
        data=full[:(640+176*320)*4];self.write('benchmark-input.f32',data);r['benchmarkInputSha256']=target.sha(data)
        for label in ['baseline','optimized']:
            for suffix in ['input','cpu','vulkan']:
                raw=next(self.source_folder.glob(f'*quality-optimized-{suffix}.f32')).read_bytes()
                self.write(f'quality-{label}-{suffix}.f32',raw)
        self.addCleanup(patch.stopall)
        patch.object(target,'reference_metrics',return_value={'stoi':.98,'siSdrDb':18.0,'snrDb':17.0}).start()

    def write(self,suffix,data):
        p=self.folder/f'{self.run_id}-{suffix}';p.write_bytes(data)
        side=p.with_suffix('.sha256') if p.suffix=='.json' else Path(str(p)+'.sha256')
        side.write_text(target.sha(data)+'\n')

    def execute(self,fixture=True):
        self.write('result.json',(json.dumps(self.result,allow_nan=True)+'\n').encode())
        return target.verify(self.folder,self.run_id,self.bundle,LIBRARY,CLEAN,True,fixture)

    def reject(self,mutate):
        mutate(self.result)
        with self.assertRaises((AssertionError,KeyError,TypeError,ValueError)):self.execute()

    def test_valid_adapted_fixture_explicitly_allowed(self):self.assertTrue(self.execute()['verified'])
    def test_adapted_fixture_never_evidence(self):
        with self.assertRaises(AssertionError):self.execute(False)
    def test_mutations(self):
        cases=[
          lambda r:r.update(success=False),lambda r:r.update(schemaVersion=2),lambda r:r.update(layers=7),
          lambda r:r.update(warmupHops=32),lambda r:r.update(caseOrder='AABB'),lambda r:r.update(elapsedMs=50000),
          lambda r:r.update(workDeadlineMs=60000),lambda r:r.update(processDeadlineMs=60000),
          lambda r:r.update(nativeLibrarySha256='0'*64),lambda r:r.update(bundleManifestSha256='0'*64),
          lambda r:r.update(baselineBundleManifestSha256='0'*64),lambda r:r['device'].update(emulatorReported=True),
          lambda r:r['device'].update(model='generic'),lambda r:r['device'].update(sdk=34),
          lambda r:r['vulkan'].update(perDispatchProfiling=True),lambda r:r['baselineVulkan'].update(perDispatchProfiling=True),
          lambda r:r['vulkan'].update(statePingPong=False),lambda r:r['vulkan'].update(gpuFiniteChecks=False),
          lambda r:r['vulkan'].update(stateCount=30),lambda r:r['vulkan'].update(precision='float16'),
          lambda r:r['vulkan'].update(driverVersion=1),lambda r:r['vulkan'].update(operations=529),
          lambda r:r['vulkan'].update(arenaBytes=100),lambda r:r['cases'][0].update(caseId='wrong'),
          lambda r:r['cases'][0].update(outputSamples=1),lambda r:r['cases'][0].update(allOutputFinite=False),
          lambda r:r['cases'][0]['rawPushMs'].__setitem__(0,float('nan')),
          lambda r:r['cases'][0]['rawPushMs'].__setitem__(0,-1),
          lambda r:r['cases'][0]['latencyMs'].update(mean=0),lambda r:r['cases'][0].update(over20Ms=0),
          lambda r:r['cases'][0].update(checksum=float('inf')),lambda r:r.update(timedFramesPerCase=True),
          lambda r:r['quality'].update(passed=False),lambda r:r['quality'].update(resetBitwiseReplay=False),
          lambda r:r['qualityBaseline'].update(partialTailsAndCancellation=False),
          lambda r:r['quality'].update(relativeRms=0),lambda r:r.update(optionalProfile={'performed':'false'}),
        ]
        pristine=copy.deepcopy(self.result)
        for i,mutate in enumerate(cases):
            with self.subTest(i=i):
                self.result=copy.deepcopy(pristine);self.reject(mutate)
    def test_watchdog_rejected(self):
        (self.folder/f'{self.run_id}-watchdog.json').write_text('{}')
        with self.assertRaises(AssertionError):self.execute()
    def test_corrupted_wave_rejected(self):
        raw=bytearray(next(self.folder.glob('*quality-optimized-vulkan.f32')).read_bytes());raw[:4]=np.float32(.5).tobytes()
        self.write('quality-optimized-vulkan.f32',raw)
        with self.assertRaises(AssertionError):self.execute()
    def test_reference_cannot_be_replaced(self):
        raw=next(self.folder.glob('*quality-optimized-vulkan.f32')).read_bytes();self.write('quality-optimized-cpu.f32',raw)
        with self.assertRaises(AssertionError):self.execute()
    def test_input_cannot_be_replaced(self):
        p=next(self.folder.glob('*benchmark-input.f32'));raw=bytearray(p.read_bytes());raw[0]^=1
        self.write('benchmark-input.f32',raw);self.result['benchmarkInputSha256']=target.sha(raw)
        with self.assertRaises(AssertionError):self.execute()
    def test_manifest_member_corruption(self):
        m=json.loads((self.bundle/'manifest.json').read_text());m['files']['plan.bin']['sha256']='0'*64
        (self.bundle/'manifest.json').write_text(json.dumps(m));self.result['bundleManifestSha256']=target.sha((self.bundle/'manifest.json').read_bytes())
        with self.assertRaises(AssertionError):self.execute()
    def test_specialized_dispatch_identity(self):
        vk={'tiledMatmulPipeline':True,'tiledConvolutionPipeline':True}
        conv=[0]*33;conv[0]=8;conv[2]=64*161;conv[6]=1;conv[7]=64;conv[8]=1;conv[9]=161
        conv[14]=conv[15]=3;conv[22]=1
        self.assertEqual(target.expected_dispatch(conv,vk),('conv_tiled',[6,4,1]))
        conv[27]=1
        self.assertEqual(target.expected_dispatch(conv,vk),('conv_3x3_tiled',[6,4,1]))
        conv[15]=1
        with self.assertRaises(AssertionError):target.expected_dispatch(conv,vk)
        conv[15]=3;conv[22]=2
        with self.assertRaises(AssertionError):target.expected_dispatch(conv,vk)
        concat=[0]*33;concat[0]=13;concat[2]=257
        self.assertEqual(target.expected_dispatch(concat,vk),('concat_cache',[3,1,1]))

    def test_optional_profile_validation(self):
        profiles=json.loads((REPO/'Validation/AndroidStreamingOptimization-vulkan-v3/emulator-download/vulkan-v3-emulator-smoke-result.json').read_text())['optionalProfile']
        vk=copy.deepcopy(self.result['vulkan']);vk['perDispatchProfiling']=True
        # This is a structural fixture, not a new measurement: update the old
        # baseline dispatch geometry to the pinned v2 plan's equivalent flattened rows.
        descriptors=target.package_identity(self.bundle,LIBRARY,True)[4]
        for profile in profiles['profiles'][:4]:
            profile['timestampValidBits']=vk['timestampValidBits']
            for op,words in zip(profile['operations'],descriptors):
                op['kernel'],op['workgroups']=target.expected_dispatch(words,vk)
        value={'performed':True,'profiledOutputBitwiseEqual':True,'frames':4,'vulkan':vk,'profiles':profiles['profiles'][:4]}
        self.result['optionalProfile']=value
        self.assertTrue(self.execute()['verified'])
        for field,value in [('profiledOutputBitwiseEqual',False),('frames',3)]:
            original=self.result['optionalProfile'][field];self.result['optionalProfile'][field]=value
            with self.assertRaises(AssertionError):self.execute()
            self.result['optionalProfile'][field]=original
        first=self.result['optionalProfile']['profiles'][0]['operations'][0]
        original=first['kernel'];first['kernel']='conv_3x3_tiled'
        with self.assertRaises(AssertionError):self.execute()
        first['kernel']=original
        original_groups=first['workgroups'];first['workgroups']=[1,1,1]
        if original_groups != [1,1,1]:
            with self.assertRaises(AssertionError):self.execute()
        first['workgroups']=original_groups
        self.result['optionalProfile']['profiles'][0]['operations'][0]['gpuMs']+=1
        with self.assertRaises(AssertionError):self.execute()

if __name__=='__main__':unittest.main()
