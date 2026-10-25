"""Local bare Git round trips only: no external writes, GPUs or model training."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3m-f-d1'
sys.path.insert(0,str(BASE/'implementation'))
from minicells_r3mf_d1 import publication as pub  # noqa: E402
from minicells_r3mf_d1 import execution as exe  # noqa: E402


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.remote = self.root/'remote.git'
        self.work = self.root/'seed'
        subprocess.run(['git','init','--bare','--quiet',str(self.remote)],check=True)
        subprocess.run(['git','init','--quiet',str(self.work)],check=True)
        self.git(self.work,'symbolic-ref','HEAD','refs/heads/main')
        self.git(self.work,'config','user.name','CPU Publication Test')
        self.git(self.work,'config','user.email','cpu@example.invalid')
        workflow = self.work/'.github/workflows/unchanged.yml'
        workflow.parent.mkdir(parents=True)
        workflow.write_text('name: Historical workflow\non: push\n')
        (self.work/'README.md').write_text('Original main\n')
        self.git(self.work,'add','.')
        self.git(self.work,'commit','-qm','main fixture')
        self.git(self.work,'remote','add','origin',str(self.remote))
        self.git(self.work,'push','-q','origin','main')
        self.parent = self.git(self.work,'rev-parse','HEAD')
        self.run = self.root/'output'
        self.fixture()
        self.addCleanup(patch.stopall)
        patch.object(pub,'REPOSITORY_URL',str(self.remote)).start()
        patch.object(pub,'_auth_env',side_effect=lambda: nullcontext(os.environ.copy())).start()

    @staticmethod
    def git(root,*args):
        return subprocess.check_output(['git',*args],cwd=root,text=True,stderr=subprocess.PIPE).strip()

    def fixture(self):
        ident = dict(experiment_id=pub.EXPERIMENT_ID,source_sha='1'*40,
                     source_code_sha256='2'*64,config_sha256='3'*64,data_sha256='4'*64,
                     sample_plan_sha256='5'*64,seed=151001301)
        manifest = dict(ident,run_id='a'*32,run_status='COMPLETE',main_or_confirmation_accessed=False,
                        arms={a:dict(status='COMPLETE',updates=200) for a in 'BCD'},
                        publication_status='PENDING_GITHUB_PREFLIGHT',remote_commit_sha=None,artifacts={})
        arms = {a:dict(identity=dict(ident,arm=a),arm=a,status='COMPLETE',logical_optimizer_updates=200,
                       curve=[{'step':300+i,'loss':float(i)} for i in range(1,201)],
                       evaluations=[]) for a in 'BCD'}
        values = {'RESULTS.json':dict(experiment_id=pub.EXPERIMENT_ID,run_id=manifest['run_id'],status='COMPLETE',arms=arms),
                  'SAMPLE_PLAN.json':{'synthetic':True},'BUDGET_PROOF.json':{'logical_updates':200},
                  'CPU_GATE_RUN_MANIFEST.json':{'synthetic':True},'A_REUSED_PREDICTIONS.json':{'synthetic':True}}
        values.update({'arm-'+a+'/ARM_RESULT.json':row for a,row in arms.items()})
        for name,value in values.items():
            path = self.run/name
            exe._atomic_json(path,value)
            manifest['artifacts'][name] = dict(bytes=path.stat().st_size,sha256=exe._file_sha(path))
        weight = self.run/'arm-B/checkpoint-step-000200.pt'
        weight.write_bytes(b'SYNTHETIC_WEIGHT_MUST_NOT_PUBLISH')
        manifest['artifacts'][weight.relative_to(self.run).as_posix()] = dict(bytes=weight.stat().st_size,sha256=exe._file_sha(weight))
        exe._atomic_json(self.run/'RUN_MANIFEST.json',manifest)

    def test_main_descended_result_only_roundtrip_and_idempotent_retry(self):
        pub.preflight(self.run)
        self.assertEqual(self.git(self.remote,'for-each-ref','--format=%(refname)','refs/heads'), 'refs/heads/main')
        receipt = pub.publish(self.run)
        self.assertEqual(receipt['status'],'PUBLISHED')
        self.assertTrue(receipt['remote_verified'])
        commit = self.git(self.remote,'rev-parse','refs/heads/'+receipt['branch'])
        self.assertEqual(commit,receipt['remote_commit_sha'])
        self.assertEqual(self.git(self.remote,'rev-parse',commit+'^'),self.parent)
        changed = self.git(self.remote,'diff','--name-only',self.parent,commit).splitlines()
        self.assertTrue(changed)
        self.assertFalse(any(p.endswith('.pt') or p.startswith('.github/') for p in changed))
        self.assertTrue(all(p.startswith(receipt['report_directory']+'/') for p in changed))
        remote_result = self.git(self.remote,'show',commit+':'+receipt['report_directory']+'/RESULTS.json')
        self.assertEqual(json.loads(remote_result),exe._read(self.run/'RESULTS.json'))
        self.assertEqual(exe._read(self.run/'RUN_MANIFEST.json')['remote_commit_sha'],commit)
        second = pub.publish(self.run)
        self.assertEqual(second['remote_commit_sha'],commit)
        self.assertTrue(second['reused_existing_remote'])
        self.assertFalse(second['push_attempted'])
        self.assertFalse(second['push_completed'])
        self.assertIsNone(second['push_exit_code'])

    def test_failed_push_retains_complete_results_and_can_retry_only_publication(self):
        original = pub._git
        def fail_push(checkout,*args,**kwargs):
            if args[0]=='push':
                raise pub.PublicationError('SIMULATED_AUTH_PUSH_FAILURE')
            return original(checkout,*args,**kwargs)
        with patch.object(pub,'_git',side_effect=fail_push):
            with self.assertRaisesRegex(pub.PublicationError,'RESULTS_RETAINED'):
                pub.publish(self.run)
        manifest = exe._read(self.run/'RUN_MANIFEST.json')
        receipt = exe._read(self.run/'PUBLICATION.json')
        self.assertEqual(manifest['run_status'],'COMPLETE')
        self.assertEqual(manifest['publication_status'],'FAILED_LOCAL_RETAINED')
        self.assertIsNone(manifest['remote_commit_sha'])
        self.assertTrue(receipt['git_commit_created'])
        self.assertTrue(receipt['push_attempted'])
        self.assertFalse(receipt['push_completed'])
        before = exe._file_sha(self.run/'RESULTS.json')
        with patch.object(exe,'advance',side_effect=AssertionError('MUST_NOT_TRAIN')):
            final = pub.publish_existing(self.run)[0]
        self.assertTrue(final['remote_verified'])
        self.assertEqual(before,exe._file_sha(self.run/'RESULTS.json'))

    def test_successful_push_without_remote_verification_is_not_success(self):
        original = pub._git
        queries = []
        def wrong_remote(checkout,*args,**kwargs):
            result = original(checkout,*args,**kwargs)
            if args[0]=='ls-remote':
                queries.append(result)
                if len(queries)==2:
                    return '0'*40+'\trefs/heads/wrong'
            return result
        with patch.object(pub,'_git',side_effect=wrong_remote):
            with self.assertRaises(pub.PublicationError):
                pub.publish(self.run)
        receipt = exe._read(self.run/'PUBLICATION.json')
        self.assertTrue(receipt['push_completed'])
        self.assertFalse(receipt['remote_verified'])
        self.assertIsNone(exe._read(self.run/'RUN_MANIFEST.json')['remote_commit_sha'])
        self.assertTrue(pub.publish(self.run)['remote_verified'])

    def test_mounted_root_is_allowed_but_symlinked_report_is_rejected(self):
        mounted = self.root/'mounted-output'
        mounted.symlink_to(self.run, target_is_directory=True)
        manifest, _, _, _ = pub.payload(mounted)
        self.assertEqual(manifest['run_id'], 'a'*32)
        report = self.run/'arm-B/ARM_RESULT.json'
        retained = report.with_name('retained.json')
        report.rename(retained)
        report.symlink_to(retained)
        with self.assertRaisesRegex(pub.PublicationError, 'SYMLINK_REJECTED'):
            pub.payload(self.run)

    def test_tampered_json_rejected_before_auth_or_remote_write(self):
        exe._atomic_json(self.run/'RESULTS.json',{'tampered':True})
        with patch.object(pub,'_auth_env',side_effect=AssertionError('MUST_NOT_AUTH')):
            with self.assertRaises(pub.PublicationError):
                pub.publish(self.run)
        self.assertFalse(exe._read(self.run/'PUBLICATION.json')['git_commit_created'])

    def test_missing_github_secret_fails_before_any_git_or_gpu_work(self):
        with patch.object(pub, '_auth_env', side_effect=RuntimeError('GITHUB_TOKEN unavailable')), patch.object(pub, '_git', side_effect=AssertionError('MUST_NOT_START_NETWORK')):
            with self.assertRaisesRegex(pub.PublicationError, 'ENABLE_EXISTING_GITHUB_TOKEN'):
                pub.preflight(self.run)
        self.assertEqual(exe._read(self.run/'RUN_MANIFEST.json')['publication_status'], 'PREFLIGHT_FAILED_NO_GPU_WORK')

    def test_network_timeout_is_bounded_and_auth_errors_are_redacted(self):
        with patch.object(pub.subprocess, 'run', side_effect=subprocess.TimeoutExpired('git', 15)):
            with self.assertRaisesRegex(pub.PublicationError, 'NETWORK_TIMEOUT_LS-REMOTE'):
                pub._git(self.run, 'ls-remote', 'origin', timeout=15)
        failed = subprocess.CompletedProcess(['git'], 1, stdout='', stderr='denied secret-value')
        with patch.object(pub.subprocess, 'run', return_value=failed):
            with self.assertRaises(pub.PublicationError) as caught:
                pub._git(self.run, 'push', 'origin', env={'GITHUB_TOKEN':'secret-value'})
        self.assertNotIn('secret-value', str(caught.exception))
        self.assertIn('[REDACTED]', str(caught.exception))

    def test_partial_failure_reports_publish_as_partial_without_checkpoints(self):
        (self.run/'RESULTS.json').unlink()
        manifest = exe._read(self.run/'RUN_MANIFEST.json')
        manifest['run_status'] = 'FAILED_RESUMABLE'
        exe._atomic_json(self.run/'RUN_MANIFEST.json',manifest)
        receipt = pub.publish(self.run)
        raw = self.git(self.remote,'show',receipt['remote_commit_sha']+':'+receipt['report_directory']+'/RUN_MANIFEST.json')
        self.assertEqual(json.loads(raw)['run_status'],'FAILED_RESUMABLE')
        names = self.git(self.remote,'ls-tree','-r','--name-only',receipt['remote_commit_sha'],receipt['report_directory']).splitlines()
        self.assertFalse(any(name.endswith('/RESULTS.json') for name in names))

    def test_publication_preflight_blocks_production_before_model_loading(self):
        # Use the frozen preflight artifacts; mock code identity only, not training or validation.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = BASE/'engineering/45c0578ffce9fc5addfee82c2f1d06d4'
            for name in ('SAMPLE_PLAN.json','BUDGET_PROOF.json','A_REUSED_PREDICTIONS.json'):
                (root/name).write_bytes((original/name).read_bytes())
            manifest = exe._read(original/'RUN_MANIFEST.json')
            manifest.update(source_sha='f'*40,source_code_sha256='current-code')
            exe._atomic_json(root/'RUN_MANIFEST.json',manifest)
            config = exe._read(BASE/'config/frozen.json')
            with patch('minicells_r3mf_d1.source_identity.verify_source_tree',return_value='f'*40), patch('minicells_r3mf_d1.source_identity.code_sha',return_value='current-code'), patch.object(pub,'preflight',side_effect=pub.PublicationError('NO_WRITE_ACCESS')), patch.object(exe,'_execute_run',side_effect=AssertionError('MUST_NOT_START_GPU')):
                with self.assertRaisesRegex(pub.PublicationError,'NO_WRITE_ACCESS'):
                    exe.run_bcd(config,root,root/'output','f'*40,'cpu')
            run_root = next((root/'output/runs').iterdir())
            retained = exe._read(run_root/'RUN_MANIFEST.json')
            retained['run_status'] = 'COMPLETE'
            retained['arms'] = {a:dict(status='COMPLETE',updates=200) for a in 'BCD'}
            values = {'arm-'+a+'/ARM_RESULT.json':dict(identity=dict(retained,arm=a),arm=a,status='COMPLETE') for a in 'BCD'}
            values['RESULTS.json'] = dict(experiment_id=pub.EXPERIMENT_ID,run_id=retained['run_id'],status='COMPLETE')
            for name,value in values.items():
                path = run_root/name
                exe._atomic_json(path,value)
                retained['artifacts'][name] = dict(bytes=path.stat().st_size,sha256=exe._file_sha(path))
            exe._atomic_json(run_root/'RUN_MANIFEST.json',retained)
            with patch('minicells_r3mf_d1.source_identity.verify_source_tree',return_value='f'*40), patch('minicells_r3mf_d1.source_identity.code_sha',return_value='current-code'), patch.object(pub,'preflight',return_value={'status':'PREFLIGHT_PASSED'}), patch.object(exe,'_execute_run',side_effect=AssertionError('MUST_NOT_RETRAIN_COMPLETE_RUN')):
                reused = exe.run_bcd(config,root,root/'output','f'*40,'cpu')
            self.assertEqual(reused['training_updates_this_launch'],0)


if __name__=='__main__':
    unittest.main()
