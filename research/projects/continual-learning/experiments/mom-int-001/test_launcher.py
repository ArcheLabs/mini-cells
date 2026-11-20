import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import kaggle_preflight as launcher


class LauncherTest(unittest.TestCase):
    def test_success_and_wheel_integrity_failure_publish(self):
        for valid in (True, False):
            with self.subTest(valid=valid), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                work = root/'working'
                work.mkdir()
                calls = []
                reports = []
                wheel = b'fixture wheel'

                def run(args, **kwargs):
                    calls.append(args)
                    if args[:2] == ['git','clone']:
                        Path(args[-1]).mkdir()
                    if 'download' in args:
                        dest = Path(args[args.index('--dest')+1])
                        (dest/'flash_linear_attention-0.2.2-py3-none-any.whl').write_bytes(wheel)
                    if any(str(arg).endswith('/preflight.py') for arg in args):
                        (Path(args[-1])/'preflight.json').write_text('{"status":"PASS"}')
                    return types.SimpleNamespace(returncode=0)

                def publish(repo, folder, *args):
                    reports.append(json.loads((folder/'decision.json').read_text()))

                mod = types.ModuleType('scripts.research.publish_experiment_results')
                mod.ExperimentSpec = lambda *args: args
                mod.load_github_token = lambda name: None
                mod.prepare_artifacts = lambda repo, name, spec, version: repo/spec[0]
                mod.push_results = publish
                import importlib.metadata
                with patch.dict(sys.modules,{mod.__name__:mod}), patch.object(sys,'path',sys.path.copy()), \
                     patch.object(launcher,'Path',side_effect=lambda x: work if x=='/kaggle/working' else Path(x)), \
                     patch.object(launcher.tempfile,'mkdtemp',return_value=td), \
                     patch.object(launcher.subprocess,'run',side_effect=run), \
                     patch.object(launcher.subprocess,'check_output',side_effect=lambda args,**kw:
                                  'source-sha' if args[0]=='git' else str(root/'snapshot')), \
                     patch.object(importlib.metadata,'version',return_value='2.0'), \
                     patch.object(launcher,'FLA_SHA256',hashlib.sha256(wheel).hexdigest() if valid else 'invalid'):
                    if valid:
                        launcher.main()
                    else:
                        with self.assertRaisesRegex(RuntimeError,'preflight incomplete'):
                            launcher.main()
                self.assertEqual(reports[0]['status'],'PASS' if valid else 'BLOCKED_OR_ERROR')
                self.assertEqual(any('install' in args for args in calls),valid)
                self.assertEqual(any(any(str(x).endswith('/preflight.py') for x in args) for args in calls),valid)
