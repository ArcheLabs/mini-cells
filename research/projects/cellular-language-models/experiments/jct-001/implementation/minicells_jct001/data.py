"""Canonical affine compositions; no language corpus and no accessible Confirmation records."""

from itertools import product
from .common import sha


def signature(path, primitives):
    a, b = 1, 0
    for index in path:
        u, v = primitives[index]
        a, b = u * a, u * b + v
    return int(a), int(b)


def equivalent_paths(record, fixture):
    return tuple(
        path
        for path in product(range(len(fixture["primitive_affines"])), repeat=fixture["path_steps"])
        if signature(path, fixture["primitive_affines"]) == (record["a"], record["b"])
    )


def reference(record):
    return record["a"] * record["x"] + record["b"]


class TaskBank:
    def __init__(self, fixture):
        self.fixture = fixture
        self._roles = {role: [] for role in ("train", "calibration", "pilot_report")}
        self.reserved_hashes = []
        signatures = sorted(
            {
                signature(path, fixture["primitive_affines"])
                for path in product(range(6), repeat=fixture["path_steps"])
            }
        )
        for a, b in signatures:
            key = sha(
                dict(version=fixture["generator_version"], salt=fixture["split_salt"], a=a, b=b)
            )
            bucket = int(key[:16], 16) % 8
            role = next(
                role for role, buckets in fixture["split_buckets"].items() if bucket in buckets
            )
            if role == "confirmation_reserved":
                self.reserved_hashes.append(key)
                continue
            for x in fixture["operands"][role]:
                record = dict(canonical_id=key, role=role, x=x, a=a, b=b)
                y = reference(record)
                low, high = fixture["classification_values"]
                if not low <= y <= high:
                    raise RuntimeError("JCT_TASK_OUTPUT_OUT_OF_FROZEN_CLASS_RANGE")
                record["target"] = int(y - low)
                self._roles[role].append(record)
        groups = {role: {r["canonical_id"] for r in rows} for role, rows in self._roles.items()}
        if any(not group for group in groups.values()):
            raise RuntimeError("JCT_EMPTY_ENGINEERING_SPLIT")
        if any(groups[a] & groups[b] for a, b in product(groups, repeat=2) if a < b):
            raise RuntimeError("JCT_CANONICAL_COMPOSITION_LEAKAGE")

    def records(self, role):
        if role not in self._roles:
            raise RuntimeError("JCT_CONFIRMATION_SEALED_OR_ROLE_UNKNOWN")
        return tuple(dict(row) for row in self._roles[role])

    def registry(self):
        return dict(
            schema="minicells.jct001.synthetic_registry.v1",
            status="ENGINEERING_IDENTITY_ONLY_NOT_SCIENTIFIC_REGISTRATION",
            generator_version=self.fixture["generator_version"],
            canonical_identity="COMPOSED_AFFINE_FUNCTION_NOT_RENDERED_STRING_OR_ARBITRARY_PATH",
            roles={
                role: dict(
                    records=len(rows),
                    canonical_compositions=len({r["canonical_id"] for r in rows}),
                    content_sha256=sha(rows),
                    canonical_ids=sorted({r["canonical_id"] for r in rows}),
                )
                for role, rows in self._roles.items()
            },
            confirmation=dict(
                status="RESERVED_ID_HASHES_ONLY_NOT_MATERIALIZED",
                records_materialized=0,
                identity_sha256=sha(sorted(self.reserved_hashes)),
                identities=len(self.reserved_hashes),
            ),
            oracle_inputs_exposed_to_router=False,
            real_story_registration=None,
            arithmetic_learnability="NOT_TESTED",
            answer_label_overlap="ALLOWED_FOR_SYNTHETIC_COMPOSITION_FIXTURE_NOT_REAL_ARITHMETIC_APPROVAL",
        )


def feature_tensor(records, device="cpu"):
    import torch

    return torch.tensor(
        [[r["x"], r["a"], r["b"]] for r in records], dtype=torch.float32, device=device
    )


def batches(data, engine, rng, device):
    import torch

    # Caller supplies only the declared TRAIN collection, never a role-selector or Oracle path.
    if data.get("role") not in ("train", "UNIT_ONLY"):
        raise RuntimeError("JCT_TRAIN_ROLE_REQUIRED")
    rows = data["records"]
    selected = [
        rows[int(rng.integers(len(rows)))] for _ in range(engine["training"]["effective_batch"])
    ]
    return [
        (
            feature_tensor(selected, device),
            torch.tensor([r["target"] for r in selected], dtype=torch.long, device=device),
        )
    ]


def unit_data(fixture):
    x, a, b = fixture["unit_update_features"]
    low, _ = fixture["classification_values"]
    return dict(
        role="UNIT_ONLY", records=[dict(x=x, a=a, b=b, target=fixture["unit_update_target"] - low)]
    )
