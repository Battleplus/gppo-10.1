"""No-model regressions: actual stdlib production contract and checkpoint merger."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import recovery_contract as rc
import task_pipeline as pipeline

ROOT=Path(__file__).resolve().parent

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.matrix=rc.read(ROOT/'experiment-matrix.json')
        self.request=rc.read(ROOT/'RESOURCE_REQUEST.json')

    def test_exact_two_training_routes(self):
        self.assertEqual(rc.training_route_keys(self.matrix),{('G1',8302),('G1',8303)})
        pipeline.validate_task_configuration(self.matrix,self.request)

    def test_all_seven_are_final_and_merged_with_two(self):
        reused=rc.reused_route_records(ROOT,self.matrix,self.request)
        self.assertEqual(set(reused),rc.REUSE_KEYS)
        with tempfile.TemporaryDirectory() as name:
            output=Path(name);routes=list(reused.values())
            for method,seed in sorted(rc.TRAIN_KEYS):
                path=output/f'{method}-{seed}.bin';path.write_bytes(b'no-model-file-identity-fixture')
                routes.append({'method':method,'seed':seed,'world_model_variant':'G1','world_model_seed':seed-100,
                               'training_steps':2048,'optimizer_updates':128,'checkpoint':str(path),'checkpoint_sha256':rc.sha(path)})
            training={'routes':routes,'checkpoints':{(r['method'],r['seed']):r['checkpoint'] for r in routes}}
            result,_=pipeline._validated_checkpoints(training,output,root=ROOT,expected_steps=2048,expected_updates=128)
            self.assertEqual(len(result),9)
            training['routes'][-1]['optimizer_updates']=28
            with self.assertRaises(pipeline.TaskPipelineError):
                pipeline._validated_checkpoints(training,output,root=ROOT,expected_steps=2048,expected_updates=128)

    def test_unapproved_mode_rejected(self):
        self.matrix['recovery']['mode']='resume_pending'
        with self.assertRaises(rc.RecoveryContractError):rc.training_route_keys(self.matrix)

    def test_scientific_change_rejected(self):
        self.matrix['prior_configuration']['G1']['prior_scale']=0.2
        with self.assertRaises(rc.RecoveryContractError):rc.verify_recovery(ROOT,self.matrix,self.request)

    def test_training_budget_wrong_rejected(self):
        self.request['stages']['conditional_policy_training']['policy_optimizer_updates']=1152
        with self.assertRaises(rc.RecoveryContractError):rc.verify_recovery(ROOT,self.matrix,self.request)

    def test_evaluation_budget_unchanged(self):
        self.request['stages']['conditional_task_confirmation']['wall_seconds']-=1
        with self.assertRaises(rc.RecoveryContractError):rc.verify_recovery(ROOT,self.matrix,self.request)

    def test_incomplete_or_pending_checkpoint_rejected(self):
        original=rc.read
        for field,value in [('ledger_pending_calls',1),('optimizer_updates',28),('ledger_completed_saves',0)]:
            c=original(ROOT/'recovery-contract.json');c['reuse_routes'][0][field]=value
            def patched(path):return c if Path(path).name=='recovery-contract.json' else original(path)
            with patch.object(rc,'read',patched),self.assertRaises(rc.RecoveryContractError):
                rc.verify_recovery(ROOT,self.matrix,self.request)

    def test_corrupt_bytes_rejected_before_model_load(self):
        real=rc.sha
        def patched(path):return '0'*64 if Path(path).suffix=='.pt' and Path(path).parent.name=='reused-policy' else real(path)
        with patch.object(rc,'sha',patched),self.assertRaises(rc.RecoveryContractError):
            rc.verify_recovery(ROOT,self.matrix,self.request)

    def test_duplicate_seed_and_world_binding_rejected(self):
        original=rc.read
        for change in ('duplicate','world'):
            c=original(ROOT/'recovery-contract.json')
            if change=='duplicate':c['reuse_routes'][1]=copy.deepcopy(c['reuse_routes'][0])
            else:next(r for r in c['reuse_routes'] if r['method']=='G1')['world_model_seed']=8202
            def patched(path):return c if Path(path).name=='recovery-contract.json' else original(path)
            with patch.object(rc,'read',patched),self.assertRaises(rc.RecoveryContractError):
                rc.verify_recovery(ROOT,self.matrix,self.request)

    def test_external_path_not_treated_as_reuse(self):
        with tempfile.TemporaryDirectory() as name:
            p=Path(name)/'external.bin';p.write_bytes(b'fixture')
            with self.assertRaises(rc.RecoveryContractError):
                rc.validate_checkpoint_location(ROOT,ROOT/'run-once',{'checkpoint':str(p),'checkpoint_sha256':rc.sha(p),'reuse':True})

if __name__=='__main__':unittest.main()
