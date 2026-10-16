import torch
from clm001_r3.model import Model
from clm001_r3.audit import capture, measure, fixed_suffix, summarize
from clm001_r3.oracle import Snapshot
from clm001_r3.probing import EventSampler
from clm001_r3.specialization import crossover, route_change
from clm001_r3.common import hash_accepted_state
from test_credit import batch


def test_actual_post_gradient_equals_exact_replay(config):
    model = Model(config, "QC-R3")
    x, y = batch()[0]
    replay = torch.tensor([[[0, 0, 4]] * 4])
    sampler = EventSampler(4, torch.Generator().manual_seed(3), 1)
    out = model(x, replay=replay, competition_sampler=sampler)
    torch.nn.functional.cross_entropy(out.logits.reshape(-1, 16), y.reshape(-1)).backward()
    snap = Snapshot(model)
    for event in sampler.events:
        s = event.state
        _, g, _ = fixed_suffix(
            snap, s, s.chosen, y[s.token_index], replay[0, s.position, s.depth + 1 :].tolist(), True
        )
        assert torch.allclose(event.gradient * 4, g, rtol=1e-5, atol=1e-7)
    sampler.close()


def test_calibration_grid_positions_coverage_and_frozen_truth(config, assets):
    folder, data = assets
    model = Model(config, "Q-R3")
    before = hash_accepted_state(model)
    bank, missing = capture(model, data, config)
    assert bank and {r["position"] for r in bank}.issuperset({0, 1})
    result = measure(model, bank[:4], config)
    assert result["records"] and all(
        "fixed_suffix_gap" in r and "long_gap" in r for r in result["records"]
    )
    assert hash_accepted_state(model) == before
    assert all(row["target"].numel() == 1 for row in bank)


def test_gate_does_not_pass_missing_evidence(config):
    result = summarize([], config)
    assert result["status"] == "INCONCLUSIVE" and result["local_agreement"]["mean"] is None


def test_crossover_requires_opposite_groups_and_multiplicity(config):
    rows = []
    for group in (0, 1):
        for story in range(4):
            rows.append(
                {
                    "cell_i": 0,
                    "cell_j": 1,
                    "group_feature": group,
                    "group": story,
                    "fixed_suffix_gap": -1.0 if group == 0 else 1.0,
                }
            )
    result = crossover(rows, config)
    assert result["candidate_count"] == 1
    assert "Bonferroni" in result["multiplicity"]
    assert not crossover([r for r in rows if r["group_feature"] == 0], config)["candidate_count"]


def test_fixed_live_route_metrics_are_distinct():
    previous = [{"cell_probabilities": [0.8, 0.2]}]
    current = [{"cell_probabilities": [0.2, 0.8]}]
    result = route_change(previous, current)
    assert result["action_flip_rate"] == 1 and result["mean_js"] > 0


def test_js_handles_inactive_reserved_slots_exactly():
    p = [{"cell_probabilities": [0.8, 0.2, 0.0]}]
    q = [{"cell_probabilities": [0.2, 0.8, 0.0]}]
    result = route_change(p, q)
    assert result["mean_js"] > 0
    assert route_change(p, p)["mean_js"] == 0.0
