"""Development-only capability, path and Cell intervention measurements."""

import copy
import hashlib
import time

import numpy as np
import torch
from torch.nn import functional as F

from minicells_r3e.common import fingerprint


def _fixed_arithmetic(records, count=32):
    selected = []
    for carry in (False, True):
        part = [r for r in records if bool(r["carry"]) == carry]
        part.sort(key=lambda r: hashlib.sha256(("R3ME-ARITH-DEV:" + r["group"]).encode()).hexdigest())
        selected.extend(part[: count // 2])
    if len(selected) != count:
        raise ValueError("ARITH_DEV_FACT_COUNT_INVALID")
    return selected


@torch.no_grad()
def measure(model, research, tokenizer, config, decode=True):
    before = fingerprint(model.state_dict())
    old_work = copy.deepcopy(model.work)
    old_training = model.training
    old_context, old_observer = model.training_context, model.observer
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    device = next(model.parameters()).device
    model.eval()
    model.training_context = None
    model.observer = None
    usage = {"story": [0] * len(model.cells), "arithmetic": [0] * len(model.cells)}
    story_loss = story_tokens = answer_loss = answer_tokens = full_loss = full_tokens = 0.0
    carry = {"carry": {"loss": 0.0, "tokens": 0, "exact": 0, "facts": 0},
             "no_carry": {"loss": 0.0, "tokens": 0, "exact": 0, "facts": 0}}
    paths = set()
    started = time.perf_counter()
    story_rows = []
    for story in research["story"]["development"]:
        loss_sum, token_count = 0.0, 0
        for ids in story["windows"]:
            x = torch.tensor([ids[:-1]], device=device, dtype=torch.long)
            y = torch.tensor(ids[1:], device=device, dtype=torch.long)
            out = model(x)
            usage["story"] = [a + int(b) for a, b in zip(usage["story"], out.operator_rows.sum(0).cpu().tolist())]
            logits = out.logits.reshape(-1, out.logits.shape[-1])
            loss = F.cross_entropy(logits, y, reduction="sum")
            story_loss += float(loss)
            story_tokens += int(y.numel())
            loss_sum += float(loss)
            token_count += int(y.numel())
            paths.update(_path_keys(out))
        story_rows.append({"group": story["group"], "nll": loss_sum / max(1, token_count), "tokens": token_count})
    arith_rows = []
    for row in _fixed_arithmetic(research["arithmetic"]["dev"], config["evaluation"]["arithmetic_groups"]):
        ids = row["ids"] + [int(config["arithmetic"]["eos_token_id"])]
        x = torch.tensor([ids[:-1]], device=device, dtype=torch.long)
        y = torch.tensor(ids[1:], device=device, dtype=torch.long)
        out = model(x)
        usage["arithmetic"] = [a + int(b) for a, b in zip(usage["arithmetic"], out.operator_rows.sum(0).cpu().tolist())]
        losses = F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]), y, reduction="none")
        answer_idx = torch.tensor([i - 1 for i in row["answer_positions"]], device=device, dtype=torch.long)
        answer_values = losses.index_select(0, answer_idx)
        full_loss += float(losses.sum())
        full_tokens += int(losses.numel())
        answer_loss += float(answer_values.sum())
        answer_tokens += int(answer_values.numel())
        paths.update(_path_keys(out))
        if decode:
            generated = _generate(
                model,
                row["prompt_ids"],
                device,
                config["evaluation"]["autoregressive_answer_token_cap"],
            )
            exact = tokenizer.decode(generated).strip() == str(row["answer"])
        else:
            exact = None
        key = "carry" if row["carry"] else "no_carry"
        carry[key]["loss"] += float(answer_values.sum())
        carry[key]["tokens"] += int(answer_values.numel())
        if exact is not None:
            carry[key]["exact"] += int(exact)
        carry[key]["facts"] += 1
        arith_rows.append({"group": row["group"], "carry": row["carry"], "answer_nll": float(answer_values.mean()), "exact_match": exact})
    if fingerprint(model.state_dict()) != before:
        raise RuntimeError("R3ME_EVALUATION_MUTATED_MODEL")
    result = {
        "story_nll": story_loss / max(1, story_tokens),
        "story_ppl": float(np.exp(min(80.0, story_loss / max(1, story_tokens)))),
        "answer_nll": answer_loss / max(1, answer_tokens),
        "full_arithmetic_nll": full_loss / max(1, full_tokens),
        "arithmetic_exact_match": (
            sum(int(r["exact_match"]) for r in arith_rows) / max(1, len(arith_rows))
            if decode
            else None
        ),
        "carry_strata": {
            k: {
                "answer_nll": v["loss"] / max(1, v["tokens"]),
                "exact_match": v["exact"] / max(1, v["facts"]) if decode else None,
                "facts": v["facts"],
            }
            for k, v in carry.items()
        },
        "teacher_forced_cell_usage": usage,
        "distinct_evaluation_paths": len(paths),
        "story_rows": story_rows,
        "arithmetic_rows": arith_rows,
        "story_role": "REUSED_HISTORICAL_DEVELOPMENT",
        "arithmetic_role": "ARITH_DEV",
        "operator_work": {k: v - old_work.get(k, 0) for k, v in model.work.items()},
        "seconds": time.perf_counter() - started,
    }
    model.work = old_work
    model.training_context, model.observer = old_context, old_observer
    model.train(old_training)
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    return result


def _path_keys(out):
    actions = getattr(out, "actions", None)
    if actions is None:
        return []
    values = actions.detach().cpu().numpy()
    if values.ndim == 2:
        values = values[:, :, None]
    if values.ndim < 2:
        return []
    return [tuple(np.asarray(path).reshape(-1).tolist()) for path in values.reshape(-1, *values.shape[-2:])]


@torch.no_grad()
def _generate(model, prompt_ids, device, cap):
    prefix = list(prompt_ids)
    generated = []
    maximum = model.c["model"]["max_sequence_length"]
    for _ in range(cap):
        ids = torch.tensor([prefix[-maximum:]], device=device, dtype=torch.long)
        token = int(model(ids).logits[0, -1].argmax())
        if token == 1:
            break
        prefix.append(token)
        generated.append(token)
    return generated


def gradient_conflict(tr, train_assets, research, seed, config):
    """Exact signed cell-gradient cosine on deterministic Train-only Story/Arithmetic probes."""
    model = tr.model
    device = next(model.parameters()).device
    story_rng = np.random.default_rng(seed + 153301 + 701)
    story_batches = __import__("minicells_r3c_moe_bridge.data", fromlist=["train_batches"]).train_batches(
        train_assets, tr.c["engine"], story_rng, device
    )
    arithmetic_rng = np.random.default_rng(seed + 153301 + 907)
    from .data import arithmetic_batches

    arithmetic_batch = arithmetic_batches(arithmetic_rng, research["arithmetic"]["train"], config, model)
    saved_rng = tr.route_rng.get_state()
    result = {}
    for task, batches in (("story", story_batches), ("arithmetic", arithmetic_batch)):
        model.zero_grad(set_to_none=True)
        values = []
        for x, y in batches:
            out = model(x, epsilon=0.0, route_rng=tr.route_rng)
            loss = F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]), y.reshape(-1))
            loss.backward()
            values.append(float(loss.detach()))
        per_cell = []
        for cell in model.cells:
            grads = [p.grad.detach().reshape(-1) for p in cell.parameters() if p.grad is not None]
            vector = torch.cat(grads).float().cpu() if grads else torch.zeros(1)
            per_cell.append(vector)
        result[task] = {"loss": sum(values) / max(1, len(values)), "gradients": per_cell}
    tr.route_rng.set_state(saved_rng)
    model.zero_grad(set_to_none=True)
    cells = []
    for i, (story_g, arith_g) in enumerate(zip(result["story"]["gradients"], result["arithmetic"]["gradients"])):
        if story_g.shape != arith_g.shape:
            raise RuntimeError("R3ME_GRADIENT_LAYOUT_CHANGED")
        denom = float(story_g.norm() * arith_g.norm())
        cosine = float(torch.dot(story_g, arith_g) / max(denom, 1e-30))
        cells.append({"cell": i, "story_norm": float(story_g.norm()), "arithmetic_norm": float(arith_g.norm()), "cosine": cosine, "signed_exact_vector_comparison": True})
    return {
        "cells": cells,
        "mean_cosine": sum(x["cosine"] for x in cells) / max(1, len(cells)),
        "sketch": "NONE_EXACT_FLATTENED_CELL_GRADIENTS",
        "probe_scope": "DETERMINISTIC_TRAIN_ONLY_NOT_PROPOSAL_OR_EVALUATION_INPUT",
        "optimizer_updates": 0,
    }


def specialization(model, research, tokenizer, config):
    """Matched-use Cell replacement tests; no development labels enter routing or training."""
    import numpy as np

    baseline = measure(model, research, tokenizer, config, decode=False)
    usage = [
        sum(baseline["teacher_forced_cell_usage"][task][i] for task in ("story", "arithmetic"))
        for i in range(len(model.cells))
    ]
    rows = []
    for cell_id in range(len(model.cells)):
        alternatives = [j for j in range(len(model.cells)) if j != cell_id]
        choices = {
            "FIXED_CYCLIC_CELL_ID": (cell_id + 1) % len(model.cells),
            "USAGE_MATCHED": min(alternatives, key=lambda j: (abs(usage[j] - usage[cell_id]), j)),
            "RANDOM_CELL_ID": int(np.random.default_rng(153301 + cell_id).choice(alternatives)),
        }
        measured = {}
        for control, replacement in choices.items():
            if replacement not in measured:
                candidate = copy.deepcopy(model)
                candidate.cells[cell_id].load_state_dict(model.cells[replacement].state_dict(), strict=True)
                measured[replacement] = measure(candidate, research, tokenizer, config, decode=False)
            result = measured[replacement]
            story_effect = (result["story_nll"] - baseline["story_nll"]) / max(baseline["story_nll"], 1e-12)
            arithmetic_effect = (result["answer_nll"] - baseline["answer_nll"]) / max(baseline["answer_nll"], 1e-12)
            rows.append(
                {
                    "cell": cell_id,
                    "replacement": replacement,
                    "story_effect": story_effect,
                    "answer_effect": arithmetic_effect,
                    "specificity": arithmetic_effect - story_effect,
                    "calls_unchanged": True,
                    "control": control,
                    "usage": usage[cell_id],
                    "replacement_usage": usage[replacement],
                    "decode_omitted_for_cost": True,
                    "scope": "LEGAL_POSTHOC_CELL_REPLACEMENT_NOT_ROUTER_INPUT",
                }
            )
    return rows


def four_grid(stable_model, reorg_model, research, tokenizer, config):
    grid = {}
    for cell_name, cell_model in (("O", stable_model), ("N", reorg_model)):
        for conn_name, conn_model in (("O", stable_model), ("N", reorg_model)):
            model = copy.deepcopy(cell_model)
            model.connection.load_state_dict(conn_model.connection.state_dict(), strict=True)
            grid[cell_name + conn_name] = measure(model, research, tokenizer, config, decode=False)
    return grid

