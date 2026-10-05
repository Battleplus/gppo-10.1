"""No environment, model initialization, checkpoint load or formal attempt."""
import hashlib
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from manifest_contract import write_identity_files,verify_package,verify_external_authorization,PackageContractError
from worker_contract import verify_worker_contract
from task_contract import verify_task_inputs

ROOT=Path(__file__).resolve().parent
class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.home=Path(self.tmp.name);self.root=self.home/'package'
        shutil.copytree(ROOT,self.root)
        self.identity=write_identity_files(self.root,attempt=json.loads((ROOT/'RESOURCE_REQUEST.json').read_text())['attempt'],entrypoint='run_g1_task_validation.py')
        self.attempt=json.loads((ROOT/'RESOURCE_REQUEST.json').read_text())['attempt']
        self.auth=self.home/'external-authorization.json'
        self.token='public-test-only-not-a-real-token'
        self.value={'schema':'w1-external-launch-authorization/2.0.0','attempt':self.attempt,
         'execution_manifest_sha256':self.identity['execution_manifest_sha256'],'hashes_sha256':self.identity['hashes_sha256'],
         'resource_request_sha256':hashlib.sha256((self.root/'RESOURCE_REQUEST.json').read_bytes()).hexdigest(),
         'status':'APPROVED','token_sha256':hashlib.sha256(self.token.encode()).hexdigest()}
    def tearDown(self):self.tmp.cleanup()
    def authwrite(self,v):self.auth.write_text(json.dumps(v)+'\n',encoding='utf-8')
    def assert_idle(self):
        for name in ['run-once','execution.lock','budget.sqlite3','authorization-consumed.json','launch-intent.json','verified-export']:
            self.assertFalse((self.root/name).exists() or (self.home/name).exists(),name)
    def test_source_contract_and_real_worker_check(self):
        verified=verify_worker_contract(self.root,self.attempt,self.identity['execution_manifest_sha256'],self.identity['hashes_sha256'])
        self.assertEqual(verified['request']['status'],'NOT_APPROVED')
        self.assertEqual(verify_task_inputs(self.root)['model_loads'],0)
        self.assert_idle()
    def test_rejections_precede_staging(self):
        variants=[('trailing',None),('identity',{**self.value,'attempt':'wrong'}),('missing',{k:v for k,v in self.value.items() if k!='attempt'}),('wrong-token',self.value)]
        for name,value in variants:
            with self.subTest(name=name):
                if value is None:self.auth.write_text(json.dumps(self.value)+' {}',encoding='utf-8')
                else:self.authwrite(value)
                with self.assertRaises(PackageContractError):
                    verify_external_authorization(self.root,self.auth,token='incorrect',preflight_only=False)
                self.assert_idle()
    def test_unapproved_structure_is_not_authorization(self):
        self.authwrite({**self.value,'status':'NOT_APPROVED','token_sha256':None})
        verify_external_authorization(self.root,self.auth,token=None,preflight_only=True)
        with self.assertRaises(PackageContractError):verify_external_authorization(self.root,self.auth,token=self.token,preflight_only=False)
        self.assert_idle()
    def test_actual_unique_entry_rejects_before_staging(self):
        variants=[('trailing',None),('identity',{**self.value,'attempt':'wrong'}),
                  ('missing',{k:v for k,v in self.value.items() if k!='attempt'}),
                  ('wrong-token',self.value)]
        for name,value in variants:
            with self.subTest(name=name):
                if value is None:self.auth.write_text(json.dumps(self.value)+' {}',encoding='utf-8')
                else:self.authwrite(value)
                result=subprocess.run([sys.executable,'-B',str(self.root/'run_g1_task_validation.py'),
                                       '--authorization-file',str(self.auth)],
                                      input='incorrect\n',capture_output=True,text=True,timeout=30)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('PackageContractError',result.stderr)
                self.assert_idle()
    def test_formal_worker_rejects_integration_before_work(self):
        p=self.root/'launch-contract.json';c=json.loads(p.read_text());c['integration_test']=True;p.write_text(json.dumps(c),encoding='utf-8')
        identity=write_identity_files(self.root,attempt=self.attempt,entrypoint='run_g1_task_validation.py')
        with self.assertRaisesRegex(RuntimeError,'FORMAL_WORKER_REJECTS_TEST_SUBSTITUTES'):
            verify_worker_contract(self.root,self.attempt,identity['execution_manifest_sha256'],identity['hashes_sha256'])
        self.assert_idle()
    def test_old_autoapproval_entry_disabled(self):
        result=subprocess.run([sys.executable,'-B',str(self.root/'run_local_research.py')],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0);self.assertIn('HISTORICAL_AUTO_APPROVAL_ENTRY_DISABLED',result.stderr)
        self.assert_idle()
    def test_inherited_launch_paths_are_disabled(self):
        names=['local_launcher.py','launch_joint_once.py','launch_once.py','wsl_stage_and_launch.py',
               'native_launch.py','acceptance_entry.py','launch_server_acceptance.py',
               'smoke_supervisor_entry.py','metered_joint_entry.py']
        for name in names:
            tree=ast.parse((self.root/name).read_text(encoding='utf-8'))
            function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='main')
            self.assertIsInstance(function.body[0],ast.Raise,name)
            self.assertIn('HISTORICAL_ENTRY_DISABLED',ast.unparse(function.body[0]))
        tree=ast.parse((self.root/'joint_supervisor_entry.py').read_text(encoding='utf-8'))
        self.assertIsInstance(tree.body[1],ast.Raise)
        result=subprocess.run([sys.executable,'-B',str(self.root/'local_launcher.py')],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('HISTORICAL_ENTRY_DISABLED',result.stderr)
        self.assert_idle()
    def test_frozen_model_byte_tampering_rejected_without_loading(self):
        path=self.root/'frozen-models/G1/seed-8201.pt'
        with path.open('ab') as f:f.write(b'x')
        with self.assertRaises(PackageContractError):verify_package(self.root)
        self.assert_idle()

if __name__=='__main__':unittest.main()
