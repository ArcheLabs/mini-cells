"""Fixed-path Cell contribution: selected residual is zeroed; routes cannot adapt."""

import math
import time
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import torch
from minicells_r3c_moe_bridge.instrumentation import MatmulAudit
from minicells_r3e.common import fingerprint
from torch.nn import functional as F


@contextmanager
def evaluation_mode(model):
    original = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(original)


def nll(logits, labels, valid):
    if (
        logits.shape[:-1] != labels.shape
        or valid.shape != labels.shape
        or not bool(((valid == 0) | (valid == 1)).all())
        or not int(valid.sum())
    ):
        raise ValueError("CLM002_VALID_TARGET_BUDGET_INVALID")
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
    ).reshape_as(labels)
    value = (losses * valid).sum() / valid.sum()
    if not torch.isfinite(value):
        raise FloatingPointError("CLM002_NONFINITE_NLL")
    return value


@torch.no_grad()
def replay(model, tokens, paths, ablated_cell=None, closed_loop=False):
    if not model.graph:
        raise ValueError("CLM002_UNAUDITED_SUBSTRATE")
    b, n = tokens.shape
    if (
        paths.shape != (b * n, model.steps)
        or paths.dtype != torch.long
        or not bool(((paths >= 0) & (paths < len(model.cells))).all())
    ):
        raise ValueError("CLM002_FIXED_PATH_INVALID")
    if ablated_cell is not None and not 0 <= ablated_cell < len(model.cells):
        raise ValueError("CLM002_CELL_ID_INVALID")
    h, memory = model.initial(tokens)
    previous = torch.full((len(h),), 6, device=h.device, dtype=torch.long)
    actions = []
    for t in range(model.steps):
        action = model.select(h, t, previous)[0] if closed_loop else paths[:, t]
        after, next_previous = model.transition(h, t, previous, action)
        # Preserve h + alpha * delta semantics exactly; no zeroing the entire residual state.
        h = (
            torch.where((action == ablated_cell)[:, None], h, after)
            if ablated_cell is not None
            else after
        )
        previous = next_previous
        actions.append(action)
    return SimpleNamespace(
        logits=model.head(h.reshape(b, n, -1)),
        actions=torch.stack(actions, -1),
        memory=memory,
        cost_semantics="ABLATION_EVALUATES_SELECTED_CELL_THEN_ZEROES_RESIDUAL; ALL_EXECUTED_CALLS_COUNT",
    )


@torch.no_grad()
def contributions(model, batches, epsilon=1e-12):
    original = fingerprint(model.state_dict())
    tasks = {}
    with evaluation_mode(model):
        for batch in batches:
            if batch["task"] not in ("Story", "Arithmetic"):
                raise ValueError("CLM002_EVALUATION_TASK_INVALID")
            x, y, valid = batch["x"], batch["y"], batch["valid"]
            weight = int(valid.sum())
            full = model(x, keep_trace=True)
            base = nll(full.logits, y, valid)
            parity = replay(model, x, full.actions)
            if not torch.allclose(full.logits, parity.logits, rtol=1e-5, atol=1e-7):
                raise RuntimeError("CLM002_FIXED_REPLAY_PARITY_FAILED")
            row = tasks.setdefault(
                batch["task"],
                {
                    "tokens": 0,
                    "loss_sum": 0.0,
                    "fixed": np.zeros(len(model.cells)),
                    "closed": np.zeros(len(model.cells)),
                    "counts": np.zeros(len(model.cells)),
                    "paths": [],
                },
            )
            row["tokens"] += weight
            row["loss_sum"] += float(base) * weight
            # Exposure is counted only on evaluated target positions, including repeated self visits.
            selected = full.actions[valid.reshape(-1).bool()]
            row["counts"] += (
                torch.bincount(selected.flatten(), minlength=len(model.cells)).cpu().numpy()
            )
            row["paths"].extend(selected.cpu().tolist())
            for cell in range(len(model.cells)):
                fixed = replay(model, x, full.actions, cell)
                closed = replay(model, x, full.actions, cell, True)
                if not torch.equal(fixed.actions, full.actions):
                    raise RuntimeError("CLM002_ABLATION_REROUTED")
                row["fixed"][cell] += float(nll(fixed.logits, y, valid)) * weight
                row["closed"][cell] += float(nll(closed.logits, y, valid)) * weight
    result = {}
    for task, row in tasks.items():
        base = row["loss_sum"] / row["tokens"]
        fixed = row["fixed"] / row["tokens"]
        closed = row["closed"] / row["tokens"]
        counts = row["counts"]
        result[task] = {
            "nll": base,
            "ppl": math.exp(base) if base < 700 else None,
            "tokens": row["tokens"],
            "fixed_nll": fixed.tolist(),
            "closed_loop_nll": closed.tolist(),
            "D": ((fixed - base) / max(abs(base), epsilon)).tolist(),
            "call_counts": counts.astype(int).tolist(),
            "call_share": (counts / max(1, counts.sum())).tolist(),
            "paths_sha256": fingerprint(row["paths"]),
        }
    if fingerprint(model.state_dict()) != original:
        raise RuntimeError("CLM002_READ_ONLY_MEASUREMENT_MUTATED_MODEL")
    return result


def specialization(tasks, minimum_share=0.01):
    if set(tasks) != {"Story", "Arithmetic"}:
        raise ValueError("CLM002_BOTH_EVALUATION_TASKS_REQUIRED")
    story, arith = tasks["Story"], tasks["Arithmetic"]
    s = np.array(story["D"]) - np.array(arith["D"])
    pos = [i for i, v in enumerate(story["call_share"]) if v >= minimum_share and s[i] > 0]
    neg = [i for i, v in enumerate(arith["call_share"]) if v >= minimum_share and s[i] < 0]
    score = min(max(s[i] for i in pos), -min(s[i] for i in neg)) if pos and neg else 0.0
    return {
        "S": float(score),
        "s": s.tolist(),
        "story_preferring_exposed_cells": pos,
        "arithmetic_preferring_exposed_cells": neg,
        "bidirectional": bool(pos and neg),
        "minimum_share": minimum_share,
        "scientific_decision": False,
    }


def js_divergence(a, b):
    a = np.array(a, dtype=float, copy=True)
    b = np.array(b, dtype=float, copy=True)
    if a.shape != b.shape or (a < 0).any() or (b < 0).any() or not a.sum() or not b.sum():
        raise ValueError("CLM002_ROUTE_DISTRIBUTION_INVALID")
    a /= a.sum()
    b /= b.sum()
    m = (a + b) / 2
    return float(sum(0.5 * np.sum(p[p > 0] * np.log(p[p > 0] / m[p > 0])) for p in (a, b)))


@torch.no_grad()
def profile_once(model, tokens, repeats=3):
    if repeats < 2:
        raise ValueError("CLM002_LATENCY_SAMPLE_COUNT")
    with evaluation_mode(model):
        with MatmulAudit(model) as meter:
            out = model(tokens)
            gemm = meter.finish(out)
        intervals = []
        for _ in range(repeats):
            if tokens.is_cuda:
                torch.cuda.synchronize(tokens.device)
            start = time.perf_counter()
            model(tokens)
            if tokens.is_cuda:
                torch.cuda.synchronize(tokens.device)
            intervals.append((time.perf_counter() - start) * 1000)
    return {
        "observed_shape_gemm": gemm,
        "p50_ms": float(np.percentile(intervals, 50)),
        "p95_ms": float(np.percentile(intervals, 95)),
        "repeats": repeats,
        "device": str(tokens.device),
        "batch": tokens.shape[0],
        "sequence": tokens.shape[1],
        "precision": str(out.logits.dtype),
        "cache": "STATIC_INPUT_MEMORY_REBUILT_EACH_FORWARD",
        "actual_cell_calls": int(out.operator_rows.sum()),
        "calls_per_token": float(out.operator_rows.sum()) / tokens.numel(),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(tokens.device)
        if tokens.is_cuda
        else None,
        "scope": "ISOLATED_INFERENCE_PROFILE_NOT_TRAINING_RESIDENT; NO_DRAM_OR_MOBILE_CLAIM",
    }


def gradient_exposure(model, batches):
    """Separate read-only checkpoint probe, never extra production backwards."""
    flags = {n: p.requires_grad for n, p in model.named_parameters()}
    original = fingerprint(model.state_dict())
    result = {}
    parameters = list(model.cells.parameters())
    try:
        for n, p in model.named_parameters():
            p.requires_grad_(n.startswith("cells."))
        with evaluation_mode(model), torch.enable_grad():
            for row in batches:
                out = model(row["x"])
                loss = nll(out.logits, row["y"], row["valid"])
                grads = torch.autograd.grad(loss, parameters, allow_unused=True)
                offset = 0
                norms = []
                for cell in model.cells:
                    length = len(list(cell.parameters()))
                    part = grads[offset : offset + length]
                    offset += length
                    values = [g.square().sum() for g in part if g is not None]
                    norms.append(float(torch.stack(values).sum().sqrt()) if values else 0.0)
                result[row["task"]] = {
                    "norms": norms,
                    "zero_gradient_cells": [i for i, n in enumerate(norms) if n == 0],
                    "call_counts": out.operator_rows.sum(0).cpu().tolist(),
                    "scope": "OFFLINE_CHECKPOINT_GRADIENT_PROBE_NO_OPTIMIZER_STEP",
                }
    finally:
        for n, p in model.named_parameters():
            p.requires_grad_(flags[n])
    if fingerprint(model.state_dict()) != original:
        raise RuntimeError("CLM002_GRADIENT_PROBE_MUTATED_STATE")
    return result


def route_switch(previous, current, previous_probe_sha, current_probe_sha):
    if previous_probe_sha != current_probe_sha or previous.shape != current.shape:
        raise ValueError("CLM002_FIXED_PROBE_IDENTITY_CHANGED")
    return {
        "action_change": float((previous != current).float().mean()),
        "path_change": float((previous != current).any(-1).float().mean()),
    }


@torch.no_grad()
def arithmetic_generation(model, questions, eos_id, max_new_tokens):
    records = []
    limit = model.c["model"]["max_sequence_length"]
    with evaluation_mode(model):
        for row in questions:
            prompt = list(row["prompt_tokens"])
            expected = list(row["answer_tokens"])
            generated = []
            tokens = prompt[:]
            if not prompt or len(prompt) + max_new_tokens > limit:
                raise ValueError("CLM002_AUTOREGRESSIVE_CONTEXT_BUDGET_EXCEEDED")
            for _ in range(max_new_tokens):
                x = torch.tensor([tokens], dtype=torch.long, device=next(model.parameters()).device)
                token = int(model(x).logits[0, -1].argmax())
                generated.append(token)
                tokens.append(token)
                if token == eos_id:
                    break
            answer = generated[:-1] if generated and generated[-1] == eos_id else generated
            records.append(
                {
                    "generated_token_ids": generated,
                    "first_token_correct": bool(
                        expected and generated and generated[0] == expected[0]
                    ),
                    "exact_match": answer == expected,
                    "eos_generated": bool(generated and generated[-1] == eos_id),
                }
            )
    return {
        "records": records,
        "exact_match": sum(r["exact_match"] for r in records) / len(records) if records else None,
        "scope": "REGISTERED_QUESTION_TOKENS_REQUIRED; NEVER_ACCESS_CONFIRMATION_FROM_ENGINEERING",
    }
