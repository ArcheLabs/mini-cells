import pytest
from minicells_clm002.statistics import factorial, stage_a, joint_decision, correct_secondary
from minicells_clm002.gates import require_scientific


def test_seed_factorial_contrast_sign_and_complete_matrix():
    rows = []
    for seed in range(5):
        losses = {
            arm + "-" + task: value
            for task in ("H", "S")
            for arm, value in [("B00", 4.0), ("B10", 3.0), ("B01", 3.5), ("B11", 2.0)]
        }
        losses["B11-S"] = 2.4
        rows.append(dict(seed=seed, losses=losses))
    report = factorial(rows)
    assert report["I_H"]["mean"] == pytest.approx(-0.5)
    assert report["delta_I"]["mean"] == pytest.approx(-0.4)
    assert report["statistical_unit"] == "TRAINING_SEED" and not report["scientific_decision"]
    assert (
        joint_decision(
            report,
            registered=True,
            stage_a_archived=True,
            integrity=True,
            quality_safe=True,
            differentiation=True,
            cost_complete=True,
            cost_regression=False,
            heterogeneity_multiplicity_pass=True,
        )
        == "JOINT_SYNERGY_SUPPORTED"
    )
    with pytest.raises(ValueError):
        factorial(rows[:1])
    assert correct_secondary([0.01, 0.03]) == pytest.approx([0.02, 0.03])


def test_stage_a_requires_confirmation_and_no_automatic_stage_b_release():
    assert stage_a([])["status"] == "STAGE_A_INCOMPLETE"
    rows = [
        dict(
            seed=s,
            complete=True,
            registered_confirmation=False,
            integrity_pass=True,
            checkpoints=[],
        )
        for s in range(5)
    ]
    assert stage_a(rows)["status"] == "STAGE_A_INCOMPLETE"
    for r in rows:
        r["registered_confirmation"] = True
        r["checkpoints"] = [
            dict(
                step=400,
                H=dict(S=0.04, bidirectional=True),
                S=dict(S=0.01),
                single_route_collapse=False,
            ),
            dict(
                step=500,
                H=dict(S=0.04, bidirectional=True),
                S=dict(S=0.01),
                single_route_collapse=False,
            ),
        ]
    result = stage_a(rows)
    assert (
        result["status"] == "STAGE_A_DIFFERENTIATION_SUPPORTED" and not result["stage_b_authorized"]
    )


@pytest.mark.parametrize("mode", ["stage-a-development", "stage-a-confirmation", "stage-b"])
def test_no_scientific_execution_from_cpu_success(config, mode):
    with pytest.raises(RuntimeError, match="GATE0_BLOCKED"):
        require_scientific(config, mode)


def test_duplicate_checkpoints_cannot_establish_durable_specialization():
    row = dict(
        seed=1,
        complete=True,
        registered_confirmation=True,
        integrity_pass=True,
        checkpoints=[
            dict(step=500, H=dict(S=0.1, bidirectional=True), S=dict(S=0)),
            dict(step=500, H=dict(S=0.1, bidirectional=True), S=dict(S=0)),
        ],
    )
    rows = [dict(row, seed=s) for s in range(5)]
    assert stage_a(rows)["status"] == "STAGE_A_INCOMPLETE"
