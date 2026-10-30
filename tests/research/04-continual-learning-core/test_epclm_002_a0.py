from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research.epclm_002_a0.runner import run_condition
from scripts.research.epclm_002_a0.run_matrix import run_matrix
from scripts.research.epclm_002_a0.task import generate_episode, generate_queries
from scripts.research.epclm_002_a0.validate import validate_protocol, validate_result


def test_protocol_is_frozen_and_seed_splits_are_valid() -> None:
    assert validate_protocol() == []


def test_writer_payload_has_no_query_and_seeded_mapping_is_reproducible() -> None:
    left = generate_episode(1234, fact_count=8, distractor_count=16)
    repeated = generate_episode(1234, fact_count=8, distractor_count=16)
    assert left == repeated
    assert "query" not in left.writer_payload()
    assert sum(event["kind"] == "distractor" for event in left.events) == 16
    assert len(left.initial_facts) == 8
    queries = generate_queries(left, seed=456)
    assert queries == generate_queries(repeated, seed=456)
    assert set(queries) == set(left.final_facts)


def test_episode_mappings_are_resampled_even_when_keys_repeat() -> None:
    seen: dict[int, set[int]] = {}
    for seed in range(400):
        episode = generate_episode(seed, fact_count=32, key_bits=8, value_bits=8)
        for key, value in episode.initial_facts.items():
            seen.setdefault(key, set()).add(value)
    assert sum(len(values) > 1 for values in seen.values()) >= 200


def test_overwrite_updates_latest_fact_and_tracks_stale_value() -> None:
    episode = generate_episode(9021, fact_count=8, overwrite_count=2)
    for key, old_value in episode.overwritten_values.items():
        assert episode.initial_facts[key] == old_value
        assert episode.final_facts[key] != old_value
        matches = [
            event["value"]
            for event in episode.events
            if event["kind"] == "fact" and event["key"] == key
        ]
        assert matches[-1] == episode.final_facts[key]


def test_learned_cell_and_flat_states_have_matched_budget() -> None:
    import torch

    from scripts.research.epclm_002_a0.data import event_tensors, query_tensor
    from scripts.research.epclm_002_a0.models import ModelConfig, build_model

    config = ModelConfig(max_events=128, cell_dim=16)
    episode = generate_episode(5542, fact_count=4, distractor_count=2)
    keys, values, is_fact, valid = event_tensors(list(episode.events))
    cells = build_model("learned-cells", config)
    cell_state = cells.encode_events(keys, values, is_fact, valid)
    logits, scores = cells(cell_state, valid, query_tensor(next(iter(episode.final_facts))))
    flat = build_model("flat-recurrent", config)
    flat_state = flat.write_events(keys, values, is_fact, valid)
    assert logits.shape == (1, 16)
    assert scores.shape[1] == config.max_events
    assert cell_state.numel() * 4 + valid.numel() == config.state_budget_bytes
    assert flat_state.numel() * 4 == config.state_budget_bytes


def test_direct_address_writer_reader_context_removal(tmp_path) -> None:
    result = run_condition(
        backend="direct-address",
        episodes=2,
        seed=80100,
        fact_count=4,
        distractor_count=16,
        negative_control_count=2,
        output=tmp_path,
    )
    path = tmp_path / "result.json"
    assert json.loads(path.read_text(encoding="utf-8"))["backend"] == "direct-address"
    assert result["metrics"]["exact_match_accuracy"] == 1.0
    assert result["metrics"]["isolation_audit_pass"] is True
    assert result["metrics"]["no_state_reader_abstained"] is True
    assert result["metrics"]["wrong_episode_negative_control_pass"] is True
    assert all(row["state_unchanged"] for row in result["records"])
    assert all(row["reader_data_allowlist_ok"] for row in result["records"])
    assert validate_result(path)["valid"] is True


def test_reader_file_guard_rejects_raw_history_open(tmp_path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw_file = raw_root / "writer-input.json"
    raw_file.write_text("{}", encoding="utf-8")
    allowed_file = tmp_path / "query.json"
    allowed_file.write_text("{}", encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["EPCLM_FORBIDDEN_ROOT"] = str(raw_root)
    script = (
        "import sys; from pathlib import Path; "
        "from scripts.research.epclm_002_a0.worker import _install_reader_io_guard; "
        "_install_reader_io_guard({Path(sys.argv[2]).resolve()}); "
        "Path(sys.argv[1]).read_text()"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(raw_file), str(allowed_file)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "raw-data root denied" in result.stderr


def test_reader_file_guard_rejects_writing_persistent_state(tmp_path) -> None:
    state_file = tmp_path / "state.json"
    state_file.write_text("original", encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    script = (
        "import sys; from pathlib import Path; "
        "from scripts.research.epclm_002_a0.worker import _install_reader_io_guard; "
        "_install_reader_io_guard({Path(sys.argv[1]).resolve()}); "
        "Path(sys.argv[1]).write_text('changed')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(state_file)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "may only read input files" in result.stderr
    assert state_file.read_text(encoding="utf-8") == "original"


def test_formal_matrix_requires_explicit_mode_and_registered_episode_count(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires --formal"):
        run_matrix(
            backend="direct-address",
            output=tmp_path,
            checkpoint_dir=None,
            episodes_per_condition=1000,
            formal=False,
        )
    with pytest.raises(ValueError, match="at least 1000"):
        run_matrix(
            backend="direct-address",
            output=tmp_path,
            checkpoint_dir=None,
            episodes_per_condition=999,
            formal=True,
        )


def test_reserved_formal_evaluation_seed_cannot_be_run_as_pilot(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires --formal"):
        run_condition(
            backend="direct-address",
            episodes=1,
            seed=98311,
            fact_count=8,
            distractor_count=0,
            output=tmp_path,
        )
