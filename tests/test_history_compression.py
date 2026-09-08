from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "research" / "history_compression_001" / "run_seed.py"
PREFLIGHT = ROOT / "scripts" / "research" / "history_compression_001" / "kaggle_preflight.py"
VISUALIZE = ROOT / "scripts" / "research" / "history_compression_001" / "visualize.py"
PROTOCOL = ROOT / "research" / "validations" / "history-compression-001" / "protocol.json"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_history_subsets_are_deterministic_and_nested() -> None:
    runner = _load(RUNNER, "hc001_runner_test")
    prompts = [f"prompt-{index}" for index in range(32)]
    indices_32, subset_32 = runner._history_subset(prompts, 26090611, 32)
    indices_8, subset_8 = runner._history_subset(prompts, 26090611, 8)
    indices_2, subset_2 = runner._history_subset(prompts, 26090611, 2)
    indices_0, subset_0 = runner._history_subset(prompts, 26090611, 0)
    assert indices_8 == indices_32[:8]
    assert indices_2 == indices_32[:2]
    assert subset_8 == subset_32[:8]
    assert subset_2 == subset_32[:2]
    assert indices_0 == []
    assert subset_0 == []
    assert runner._history_subset(prompts, 26090611, 8) == (indices_8, subset_8)


def test_frozen_protocol_has_expected_budget_ladder_and_disjoint_history() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["experiment"] == "HISTORY_COMPRESSION_001"
    assert protocol["formal_seeds"] == [26090611, 26090612, 26090613]
    modes = protocol["compression_modes"]
    assert [mode["id"] for mode in modes] == ["full_32", "tiny_8", "tiny_2", "zero_0"]
    assert [mode["history_prompt_count"] for mode in modes] == [32, 8, 2, 0]
    selection = protocol["history"]["selection_prompts"]
    evaluation = protocol["history"]["evaluation_prompts"]
    assert len(selection) == 32
    assert len(evaluation) == 32
    assert not (set(selection) & set(evaluation))
    assert protocol["mutation"]["group_size"] / protocol["mutation"]["expected_intermediate_size"] == 0.0625


def test_preflight_accepts_canonical_and_legacy_memory_flags() -> None:
    preflight = _load(PREFLIGHT, "hc001_preflight_test")
    canonical = preflight.build_parser().parse_args(["--minimum-free-mb", "12345"])
    legacy = preflight.build_parser().parse_args(["--min-free-mib", "12345"])
    assert canonical.minimum_free_mb == 12345
    assert legacy.minimum_free_mb == 12345


def test_visualization_svg_is_self_contained() -> None:
    visualize = _load(VISUALIZE, "hc001_visualize_test")
    summary = [
        {
            "mode": "zero_0",
            "history_prompt_count": 0,
            "pass_count": 1,
            "seed_count": 3,
            "median_gain": 3.0,
            "median_kl": 0.08,
            "median_top1": 0.95,
            "coordinates": [(0, 5)],
        },
        {
            "mode": "full_32",
            "history_prompt_count": 32,
            "pass_count": 3,
            "seed_count": 3,
            "median_gain": 12.0,
            "median_kl": 0.001,
            "median_top1": 1.0,
            "coordinates": [(0, 5)],
        },
    ]
    svg = visualize._svg(summary, {"status": "TEST", "minimum_observed_supported_history_prompts": 32})
    assert svg.startswith("<svg")
    assert "History Compression 001" in svg
    assert "full_32=E0/G5" in svg
    assert "<script" not in svg
