"""Observe original Engine commits without changing training or admission decisions."""

import copy

import torch
from interpolation_audit import Engine
from minicells_r3e.common import fingerprint


def receipt(before, after, proposal, package=None):
    first, final = fingerprint(before), fingerprint(after)
    accepted = bool(proposal and proposal["accepted"])
    if not accepted:
        if first != final:
            raise RuntimeError("REJECTED_CONNECTION_CHANGED")
        expected = first
    else:
        if package is None or proposal["scale"] <= 0:
            raise RuntimeError("ACCEPTED_CONNECTION_MISSING_PACKAGE")
        if fingerprint(package["original_connection"]) != first:
            raise RuntimeError("CANDIDATE_ORIGIN_MISMATCH")
        expected_values = copy.deepcopy(before)
        for k, value in before.items():
            expected_values[k] = value + proposal["scale"] * (
                package["candidate_connection"][k].to(value) - value
            )
        expected = fingerprint(expected_values)
        if final != expected or final != proposal["selected_connection_sha256"] or final == first:
            raise RuntimeError("ACCEPTED_CONNECTION_COMMIT_MISMATCH")
    return {
        "before_sha256": first,
        "after_sha256": final,
        "expected_selected_sha256": expected,
        "accepted": accepted,
        "scale": proposal["scale"] if proposal else None,
        "outer_engine_connection_commits": int(accepted),
        "scope": "ENGINEERING_OUTER_ENGINE_NOT_FORMAL_SCIENTIFIC_APPROVAL",
        "scan_formal_connection_commits": 0,
        "scan_counter_scope": "SCAN_ONLY_DOES_NOT_MUTATE_OFFICIAL_MODEL",
        "parameter_consistency": "PASS",
    }


class VerifiedEngine(Engine):
    def update(self, stream, checkpoint, references, failure_hook=None):
        before = copy.deepcopy(self.model.connection.state_dict())

        def verify(tr):
            proposal = tr.curve[-1]["proposal"]
            package = None
            if proposal and proposal["accepted"]:
                package = torch.load(
                    tr.stage_directory / "proposals" / str(tr.step) / "candidate.pt",
                    map_location="cpu",
                    weights_only=False,
                )
            record = receipt(before, tr.model.connection.state_dict(), proposal, package)
            tr.curve[-1]["connection_commit_receipt"] = record
            if failure_hook:
                failure_hook(tr)
            # A caller hook is not allowed to silently change the committed Connection either.
            if fingerprint(tr.model.connection.state_dict()) != record["after_sha256"]:
                raise RuntimeError("POST_COMMIT_HOOK_MUTATED_CONNECTION")

        result = super().update(stream, checkpoint, references, verify)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        values = copy.deepcopy(self.model.connection.state_dict())
        for k in values:
            values[k] = saved["state"]["model"]["connection." + k]
        if fingerprint(values) != self.curve[-1]["connection_commit_receipt"]["after_sha256"]:
            raise RuntimeError("SAVED_CONNECTION_COMMIT_MISMATCH")
        return result
