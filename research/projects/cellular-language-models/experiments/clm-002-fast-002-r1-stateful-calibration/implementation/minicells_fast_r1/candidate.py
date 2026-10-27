"""The frozen R3e all-action Q / eight mean-loss AdamW rounds, without admission shortcuts."""

import copy

import torch
from minicells_jct001.connection import TorchAdapter
from minicells_r3e import policy_proposal
from minicells_r3e.common import fingerprint
from minicells_r3k.reuse import bind

from .common import atomic, digest, read


def generate(engine, journal, directory, identity):
    if (
        not getattr(engine, "r1_cpu_fixture", False)
        and not engine.fast["diagnostic"]["gate1_execution_authorized"]
    ):
        raise RuntimeError("R1_REAL_CANDIDATE_SEARCH_REQUIRES_EXPLICIT_GATE1_APPROVAL")
    artifact = directory / "candidate.pt"
    receipt = directory / "CANDIDATE.json"
    if receipt.exists():
        meta = read(receipt)
        if meta["identity"] != identity or digest(artifact) != meta["sha256"]:
            raise RuntimeError("R1_CANDIDATE_IDENTITY_OR_BYTES_CHANGED")
        if (
            journal.summary()["uncertain_optimizer_calls"]
            or journal.watermark() != meta["watermark"]
        ):
            raise RuntimeError("R1_UNCERTAIN_CANDIDATE_OPTIMIZER_NO_AUTOMATIC_RETRY")
        return torch.load(
            artifact, map_location=next(engine.model.parameters()).device, weights_only=False
        )
    if len(journal.events) != 1:
        raise RuntimeError("R1_INCOMPLETE_CANDIDATE_SEARCH_NO_AUTOMATIC_RETRY")
    m, c = engine.model, engine.c
    original = fingerprint(m.state_dict())
    work = copy.deepcopy(m.work)
    cf_before = engine.cf_rng.get_state()
    collector = torch.no_grad()(
        bind(policy_proposal.collect.__wrapped__, train_batches=lambda *args: engine.pending_batch)
    )
    # Preserve FAST001's registered treatment of finite zero-gradient attempts.
    journal.allow_zero_signal = engine.journal.allow_zero_signal
    adapter = TorchAdapter(journal)
    with journal.phase("candidate_generation_train_only"):
        records = collector(engine, {})
        if len(records) != c["training"]["states"]:
            raise RuntimeError("R1_REGISTERED_Q_STATE_COUNT_CHANGED")
        journal.append(
            "AUXILIARY_LABEL_WORK", {"target_evaluations": len(records) * (6 if m.graph else 15)}
        )
        candidate = copy.deepcopy(m)
        for p in candidate.parameters():
            p.requires_grad_(False)
        for p in candidate.connection.parameters():
            p.requires_grad_(True)
        opt = adapter.optim.AdamW(
            candidate.connection.parameters(),
            lr=c["training"]["connection_lr"],
            weight_decay=c["training"]["connection_weight_decay"],
        )
        initial = copy.deepcopy(opt.state_dict())
        losses = []
        for _ in range(c["training"]["policy_steps"]):
            opt.zero_grad(set_to_none=True)
            terms = []
            for row in records:
                q, _, _ = candidate.scores(row["h"], row["stage"], row["previous"])
                costs = row["values"] - row["values"].mean()
                terms.append((q.softmax(-1) * costs).sum())
            loss = torch.stack(terms).mean()
            loss.backward()
            grads = [p.grad for p in candidate.connection.parameters() if p.grad is not None]
            if (
                not torch.isfinite(loss)
                or not grads
                or not all(torch.isfinite(g).all() for g in grads)
            ):
                raise FloatingPointError("R1_INVALID_POLICY_GRADIENT")
            torch.nn.utils.clip_grad_norm_(
                candidate.connection.parameters(), c["training"]["connection_clip"]
            )
            opt.step()
            losses.append(float(loss.detach()))
        payload = {
            "identity": identity,
            "original_connection": copy.deepcopy(m.connection.state_dict()),
            "candidate_connection": copy.deepcopy(candidate.connection.state_dict()),
            "optimizer_initial": initial,
            "optimizer_final": opt.state_dict(),
            "cf_before": cf_before,
            "cf_after": engine.cf_rng.get_state(),
            "losses": losses,
            "states": len(records),
            "actions": 6 if m.graph else 15,
            "train_input_sha256": fingerprint(engine.pending_batch),
            "source_model_sha256": original,
        }
    m.work = work
    if fingerprint(m.state_dict()) != original:
        raise RuntimeError("R1_CANDIDATE_SEARCH_MUTATED_OFFICIAL_WEIGHTS")
    if (
        journal.summary()["optimizer_calls_by_group"].get("connection", 0)
        != c["training"]["policy_steps"]
    ):
        raise RuntimeError("R1_CANDIDATE_OPTIMIZER_ACCOUNTING_MISMATCH")
    torch.save(payload, artifact)
    atomic(
        receipt,
        {
            "identity": identity,
            "sha256": digest(artifact),
            "watermark": journal.watermark(),
            "actual_optimizer_calls": len(losses),
            "source_model_sha256": original,
            "formal_connection_commits": 0,
            "confirm_read": False,
        },
    )
    return payload
