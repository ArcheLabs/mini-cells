import json
from pathlib import Path
import subprocess

import pytest


def test_default_workload_has_no_runtime_supervisor():
    from pd001.configs import ROOT
    base = ROOT / 'research/projects/cellular-language-models/experiments/006-path-first-clm/extension-parameter-depth'
    for name in ('performance-notebook.ipynb', 'notebook.ipynb'):
        notebook = json.loads((base / name).read_text())
        code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
        for forbidden in ('killpg', 'SIGTERM', 'SIGKILL', 'SessionBudget', 'BUDGET.run', '7200', 'timeout='):
            assert forbidden not in code
    notebook = json.loads((base / 'performance-notebook.ipynb').read_text())
    code = ''.join(notebook['cells'][1]['source'])
    assert "CASES = ['B0','LIGHT']" in code
    assert 'WARMUP, STEPS, REPEATS = 1, 5, 1' in code
    training = (base/'implementation/pd001/training.py').read_text()
    audit = (base/'implementation/pd001/perf_audit.py').read_text()
    assert 'pause_due' not in training
    for forbidden in ('session_deadline', 'process.kill', 'process.terminate', 'PERFORMANCE_TIME_LIMIT'):
        assert forbidden not in audit
    assert not (base/'implementation/pd001/session_budget.py').exists()


def test_failed_benchmark_retains_completed_rows(tmp_path, monkeypatch, proposal):
    from argparse import Namespace
    from pd001 import perf_audit

    monkeypatch.setattr(perf_audit.subprocess, 'check_output',
                        lambda command, **kwargs: perf_audit.BASELINE if 'rev-parse' in command else '')
    calls = []

    def worker(command, **kwargs):
        assert "timeout" not in kwargs
        calls.append(command)
        result = Path(command[command.index('--result')+1])
        if len(calls) == 1:
            result.write_text(json.dumps({'arm': 'M0', 'case': 'B0', 'optimizer_step_ms_p50': 1}))
            return subprocess.CompletedProcess(command, 0)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(perf_audit.subprocess, 'run', worker)
    args = Namespace(steps=1, warmup=0, repeats=1, device='cpu', all_arms=False, arm='M0',
                     baseline_root=str(tmp_path), output=str(tmp_path/'evidence'),
                     cases=['B0', 'LIGHT'], perf_fixture=True)
    with pytest.raises(subprocess.CalledProcessError):
        perf_audit.run(proposal, args)
    report = json.loads((tmp_path/'evidence/ablation.json').read_text())
    assert report['status'] == 'PERFORMANCE_FAILED'
    assert len(report['rows']) == 1 and report['partial']
    assert report['READY_FOR_RETRAIN'] == 'NO'
