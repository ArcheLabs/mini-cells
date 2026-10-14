import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from pd001 import session_budget as budget


def test_deadline_is_shared_and_cannot_be_extended(monkeypatch):
    monkeypatch.delenv(budget.DEADLINE_ENV, raising=False)
    first = budget.session_deadline()
    assert budget.session_deadline() == first
    monkeypatch.setenv(budget.DEADLINE_ENV, str(time.monotonic() + 100000))
    assert budget.session_deadline() <= time.monotonic() + 7200


def test_pause_uses_session_not_training_start(monkeypatch):
    monkeypatch.setenv(budget.DEADLINE_ENV, str(time.monotonic() + 599))
    assert budget.pause_due()
    monkeypatch.setenv(budget.DEADLINE_ENV, str(time.monotonic() + 601))
    assert not budget.pause_due()


def test_timeout_kills_entire_group_and_preserves_evidence(tmp_path, monkeypatch):
    # Child ignores TERM, including after its parent has exited. Group KILL must reach it.
    monkeypatch.setenv(budget.DEADLINE_ENV, str(time.monotonic() + 0.4))
    monkeypatch.setattr(budget, 'TERMINATION_RESERVE_SECONDS', 0)
    status = tmp_path / 'status.json'
    pidfile = tmp_path / 'child.pid'
    child_code = ('import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                  f'open({str(pidfile)!r},"w").write(str(os.getpid())); time.sleep(60)')
    parent_code = f'import subprocess,time; subprocess.Popen([{sys.executable!r},"-c",{child_code!r}]); time.sleep(60)'
    supervisor = budget.SessionBudget(status)
    with pytest.raises(subprocess.TimeoutExpired):
        supervisor.run([sys.executable, '-c', parent_code])
    assert json.loads(status.read_text())['status'] == 'SESSION_TIME_LIMIT'
    pid = int(pidfile.read_text())
    # A zombie has stopped computing; its unrelated reaper owns cleanup.
    stat = Path(f'/proc/{pid}/stat')
    for _ in range(50):
        if not stat.exists() or stat.read_text().split()[2] == 'Z':
            break
        time.sleep(.01)
    else:
        os.kill(pid, 9)
        pytest.fail('GPU child could survive hard stop')


def test_notebooks_embed_shared_supervisor_and_short_defaults():
    from pd001.configs import ROOT
    base = ROOT / 'research/projects/cellular-language-models/experiments/006-path-first-clm/extension-parameter-depth'
    module = Path(budget.__file__).read_text()
    for name in ('performance-notebook.ipynb', 'notebook.ipynb'):
        notebook = json.loads((base / name).read_text())
        code = ''.join(notebook['cells'][1]['source'])
        assert code.startswith(module)
        assert 'BUDGET = SessionBudget' in code
    code = ''.join(notebook['cells'][1]['source'])
    perf = json.loads((base / 'performance-notebook.ipynb').read_text())
    code = ''.join(perf['cells'][1]['source'])
    assert "CASES = ['B0','LIGHT']" in code
    assert 'WARMUP, STEPS, REPEATS = 1, 5, 1' in code


def test_failed_benchmark_retains_completed_rows(tmp_path, monkeypatch, proposal):
    from argparse import Namespace
    from pd001 import perf_audit

    monkeypatch.delenv(budget.DEADLINE_ENV, raising=False)
    monkeypatch.setattr(perf_audit.subprocess, 'check_output',
                        lambda command, **kwargs: perf_audit.BASELINE if 'rev-parse' in command else '')
    calls = []

    class Worker:
        def __init__(self, command, **kwargs):
            calls.append(command)
            self.result = Path(command[command.index('--result')+1])

        def wait(self, **kwargs):
            if len(calls) == 1:
                self.result.write_text(json.dumps({'arm': 'M0', 'case': 'B0', 'optimizer_step_ms_p50': 1}))
                return 0
            return 1

    monkeypatch.setattr(perf_audit.subprocess, 'Popen', Worker)
    args = Namespace(steps=1, warmup=0, repeats=1, device='cpu', all_arms=False, arm='M0',
                     baseline_root=str(tmp_path), output=str(tmp_path/'evidence'),
                     cases=['B0', 'LIGHT'], perf_fixture=True)
    with pytest.raises(subprocess.CalledProcessError):
        perf_audit.run(proposal, args)
    report = json.loads((tmp_path/'evidence/ablation.json').read_text())
    assert report['status'] == 'PERFORMANCE_FAILED'
    assert len(report['rows']) == 1 and report['partial']
    assert report['READY_FOR_RETRAIN'] == 'NO'
