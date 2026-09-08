"""Guards for the CUDA execution-only repair of PCU-LOCALITY-WIDTH-001."""

from __future__ import annotations

import inspect
from pathlib import Path

from minicells.pcu_kill_001 import locality_width_isolated


def test_scientific_core_is_pinned_to_original_locality_width_source() -> None:
    assert locality_width_isolated.SCIENTIFIC_SOURCE_COMMIT == "a567c3d386ebbbcc1b5707be4af69fedd27fb455"
    assert locality_width_isolated.SCIENTIFIC_CORE_BLOB_SHA == "3e0528380baa4b9dba0d5fe51871f1f98a578264"
    assert locality_width_isolated.EXECUTION_MODE == "spawned_python_process_per_cuda_device"


def test_isolated_orchestrator_uses_subprocesses_not_cuda_threads() -> None:
    source = inspect.getsource(locality_width_isolated)
    orchestrator = inspect.getsource(locality_width_isolated.run_locality_width_diagnostic_isolated)
    worker = inspect.getsource(locality_width_isolated.run_width_worker)
    assert "ThreadPoolExecutor" not in source
    assert "subprocess.Popen" in source
    assert "run_pcu_locality_width_worker.py" in source
    assert "_run_one_width" in worker
    assert "torch.cuda.set_device" in worker
    assert "scientific_semantics_changed\": False" in source
    assert "_assert_scientific_core_unchanged" in orchestrator


def test_existing_width_results_keep_original_scientific_identity() -> None:
    source = inspect.getsource(locality_width_isolated.run_width_worker)
    assert "scientific_source = _scientific_source()" in source
    assert "source=scientific_source" in source
    assert "execution_isolation" in source


def test_canonical_runner_uses_isolated_orchestrator() -> None:
    root = Path(__file__).resolve().parents[3]
    runner = (root / "scripts/research/run_pcu_locality_width_001.py").read_text(encoding="utf-8")
    assert "run_locality_width_diagnostic_isolated" in runner
    assert "run_locality_width_diagnostic(" not in runner


def test_cuda_repair_has_no_formal_path() -> None:
    source = inspect.getsource(locality_width_isolated)
    assert "run_formal" not in source
    assert "mark_formal_seed" not in source
    assert '"formal_execution_not_started": True' in source
    assert '"scientific_evidence": False' in source
