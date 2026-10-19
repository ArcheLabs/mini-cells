from pathlib import Path
from minicells_runtime.store import canonical, digest, read, ContinuationError

BASE = Path(__file__).resolve().parents[2]
ENGINE_SHA = "ed2accbc90f541325ca2f6c53229e9c81475768d"
TRAINING_SHA = "a7792ef30582dfcaf324cca7e5245c8430df48b2"
RUN_ID = "r3b-a7792ef30582-ea3ea51c"
EXPERIMENT = "CLM-001-R3b-CF-001"
RESULT_COMMIT = "f65a7401538e8092b763e689ba7fc9358458b825"
RESULT_RELATIVE = (
    "research/projects/cellular-language-models/experiments/clm-001-qfirst-integrated/r3b/artifacts/"
    + TRAINING_SHA
    + "/536fad45745b"
)
POLICIES = (
    "BASE_GREEDY",
    "NOOP_CONTROL",
    "FORCE_STOP_D4",
    "FORCE_STOP_D8",
    "FORCE_STOP_D12",
    "CTX_ONCE",
    "NO_ADJACENT_SAME_CELL",
    "CTX_ONCE_AND_NO_REPEAT",
)


def configuration():
    c = read(BASE / "config/frozen-cf-v1.json")
    if (
        c["experiment_id"] != EXPERIMENT
        or c["training_source_sha"] != TRAINING_SHA
        or c["engine_sha"] != ENGINE_SHA
    ):
        raise ContinuationError("CF_PROTOCOL_IDENTITY_MISMATCH")
    if c["force_stop_depths"] != [4, 8, 12] or c["lambda_cost"] != 0.01 or c["mc_samples"] != 8:
        raise ContinuationError("CF_PROTOCOL_CHANGED")
    return c


def protocol_identity(c):
    return canonical(
        {
            "configuration": c,
            "spec_sha256": digest(BASE / "SPEC.md"),
            "freeze_sha256": digest(BASE / "FREEZE.md"),
        }
    )
