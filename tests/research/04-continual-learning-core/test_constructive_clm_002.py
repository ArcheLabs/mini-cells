from __future__ import annotations

import inspect

from minicells.constructive_clm_002 import StreamingGrowthLearner, run_seed


def test_constructive_clm_002_development_seed_tracks_latent_growth() -> None:
    result = run_seed(301)

    assert result["pass"] is True
    assert [row["transactions"] for row in result["checkpoints"]] == [
        256,
        512,
        1024,
        2048,
        4096,
    ]
    assert [row["true_factors"] for row in result["checkpoints"]] == [9, 12, 16, 21, 30]
    assert [row["cells"] for row in result["checkpoints"]] == [9, 12, 16, 21, 30]
    assert result["hard_cell_cap"] is None
    assert result["last_spawn_step"] >= int(0.90 * 4096)
    assert result["transaction_to_cell_compression"] >= 100.0
    assert all(result["gates"].values())


def test_streaming_growth_learner_never_accepts_hidden_factor_or_novelty_labels() -> None:
    parameters = list(inspect.signature(StreamingGrowthLearner.observe).parameters)
    assert parameters == ["self", "x", "y", "step"]
    assert "factor" not in parameters
    assert "novel" not in parameters
