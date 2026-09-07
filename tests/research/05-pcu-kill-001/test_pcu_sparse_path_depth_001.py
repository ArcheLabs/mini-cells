"""Static/unit guards for PCU-SPARSE-PATH-DEPTH-001."""

from __future__ import annotations

import inspect
from pathlib import Path

from minicells.pcu_kill_001 import sparse_path_depth


def test_canonical_topologies_are_nested() -> None:
    specs = sparse_path_depth.choose_nested_topologies(tuple(range(24)))
    assert specs[3].layers == (7, 15, 23)
    assert specs[4].layers == (7, 11, 15, 23)
    assert specs[5].layers == (7, 11, 15, 19, 23)
    assert set(specs[3].layers).issubset(specs[4].layers)
    assert set(specs[4].layers).issubset(specs[5].layers)


def test_all_depths_have_equal_added_cell_and_step_budget() -> None:
    specs = sparse_path_depth.choose_nested_topologies(tuple(range(24)))
    assert sparse_path_depth.TOTAL_ADDED_K == 32
    assert sparse_path_depth.TOTAL_ADDED_STEPS == 256
    for depth in sparse_path_depth.DEPTHS:
        spec = specs[depth]
        assert sum(spec.transport_k) == 16
        assert sum(spec.transport_steps) == 128
        assert spec.readout_k == 16
        assert spec.readout_steps == 128
        assert sum(spec.transport_k) + spec.readout_k == 32
        assert sum(spec.transport_steps) + spec.readout_steps == 256
    assert specs[3].transport_k == (16,)
    assert specs[3].transport_steps == (128,)
    assert specs[4].transport_k == (8, 8)
    assert specs[4].transport_steps == (64, 64)
    assert specs[5].transport_k == (6, 5, 5)
    assert specs[5].transport_steps == (43, 43, 42)


def test_each_added_layer_is_allocated_then_trained_then_frozen() -> None:
    source = inspect.getsource(sparse_path_depth._train_one_added_layer)
    assert "full_model_task_conditioned_allocation(" in source
    assert "_train_full_model_branch(" in source
    assert source.index("full_model_task_conditioned_allocation(") < source.index("_train_full_model_branch(")
    assert "runtime.requires_grad_(False)" in source
    assert "model.requires_grad_(False)" in source
    assert "first64_A_train_answer_CE_gradient_under_preceding_frozen_path" in source


def test_l7_hybrid_is_exactly_replayed_and_frozen_first() -> None:
    source = inspect.getsource(sparse_path_depth.run_topology)
    assert "_train_hybrid_branch(" in source
    assert "SPARSE_PATH_L7_REPRODUCTION_MISMATCH ranking" in source
    assert "SPARSE_PATH_L7_REPRODUCTION_MISMATCH direct" in source
    assert "_freeze_l7_runtime(model, runtime7)" in source
    assert source.index("_freeze_l7_runtime(model, runtime7)") < source.index("for layer, k, steps in zip")


def test_depth_sweep_requires_all_three_depths_and_both_gates_for_rescue() -> None:
    def row(direct: float, ranking: float) -> dict:
        return {"metrics": {"direct_accuracy": direct, "ranking_eval_accuracy": ranking}}

    assert sparse_path_depth.classify_depth_sweep({
        3: row(0.81, 0.81),
        4: row(0.20, 0.90),
        5: row(0.30, 0.90),
    }) == "SPARSE_PATH_DEPTH_3_RESCUES_NATIVE_GENERATION"
    assert sparse_path_depth.classify_depth_sweep({
        3: row(0.20, 0.90),
        4: row(0.82, 0.82),
        5: row(0.90, 0.90),
    }) == "SPARSE_PATH_DEPTH_4_RESCUES_NATIVE_GENERATION"
    assert sparse_path_depth.classify_depth_sweep({
        3: row(0.20, 0.90),
        4: row(0.30, 0.90),
        5: row(0.79, 0.90),
    }) == "DEEPER_SPARSE_PATH_IMPROVES_BUT_DOES_NOT_RESCUE"


def test_experiment_is_engineering_only() -> None:
    source = inspect.getsource(sparse_path_depth)
    assert "run_formal" not in source
    assert "mark_formal_seed" not in source
    assert '"scientific_evidence": False' in source
    assert '"formal_execution_not_started": True' in source


def test_dual_gpu_runner_uses_process_isolation_not_threads() -> None:
    runner = Path(__file__).resolve().parents[3] / "scripts/research/run_pcu_sparse_path_depth_001.py"
    source = runner.read_text(encoding="utf-8")
    assert "subprocess.Popen" in source
    assert "ThreadPoolExecutor" not in source
    assert '(3, "cuda:0"' in source
    assert '(4, "cuda:1"' in source
    assert '(5, "cuda:0"' in source
    assert "torch.cuda.device_count() < 2" in source
