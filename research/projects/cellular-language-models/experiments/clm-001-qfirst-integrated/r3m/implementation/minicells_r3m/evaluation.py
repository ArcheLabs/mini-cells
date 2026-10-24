"""Read-only Story/answer evaluation, exact-match generation and symmetric four-grid interventions."""

import copy
import time
import hashlib
import torch
from torch.nn import functional as F
from minicells_r3e.common import fingerprint


def subset(records, groups=32):
    selected = []
    for carry in (False, True):
        names = sorted(
            {r["group"] for r in records if r["carry"] == carry},
            key=lambda g: hashlib.sha256(g.encode()).hexdigest(),
        )
        selected.extend(names[: groups // 2])
    return [r for r in records if r["group"] in selected]


@torch.no_grad()
def measure(model, research, role, tokenizer=None):
    if role not in ("trigger", "pilot_report", "main_report"):
        raise RuntimeError("EVALUATION_ROLE_FORBIDDEN")
    before = fingerprint(model.state_dict())
    before_work = copy.deepcopy(model.work)
    usage = {r: [0] * len(model.cells) for r in ("story", "arithmetic")}
    model.eval()
    model.training_context = None
    device = next(model.parameters()).device
    sums = dict(
        story_loss=0.0,
        story_tokens=0,
        answer_loss=0.0,
        answer_tokens=0,
        full_loss=0.0,
        full_tokens=0,
        exact=0,
        problems=0,
    )
    rows = []
    begin = time.perf_counter()
    for story in research["story"][role][:32]:
        losses = []
        for ids in story["windows"][:2]:
            x = torch.tensor([ids[:-1]], device=device)
            y = torch.tensor(ids[1:], device=device)
            value = model(x)
            usage["story"] = [
                a + int(b)
                for a, b in zip(usage["story"], value.operator_rows.sum(0).cpu().tolist())
            ]
            out = value.logits
            loss = F.cross_entropy(out.reshape(-1, out.shape[-1]), y, reduction="sum")
            sums["story_loss"] += float(loss)
            sums["story_tokens"] += y.numel()
            losses.append(float(loss) / y.numel())
        rows.append(dict(task="story", group=story["group"], nll=sum(losses) / len(losses)))
    for problem in subset(research["arithmetic"][role]):
        ids = problem["ids"]
        x = torch.tensor([ids[:-1]], device=device)
        y = torch.tensor(ids[1:], device=device)
        value = model(x)
        usage["arithmetic"] = [
            a + int(b)
            for a, b in zip(usage["arithmetic"], value.operator_rows.sum(0).cpu().tolist())
        ]
        out = value.logits
        losses = F.cross_entropy(out.reshape(-1, out.shape[-1]), y, reduction="none")
        mask = torch.tensor([i - 1 for i in problem["answer_positions"]], device=device)
        answer = losses.index_select(0, mask)
        sums["full_loss"] += float(losses.sum())
        sums["full_tokens"] += len(losses)
        sums["answer_loss"] += float(answer.sum())
        sums["answer_tokens"] += len(answer)
        prefix = list(problem["prompt_ids"])
        generated = []
        # Fixed cap is frozen, independent of the expected answer length.
        for _ in range(8):
            value = int(
                model(
                    torch.tensor(
                        [prefix[-model.c["model"]["max_sequence_length"] :]], device=device
                    )
                )
                .logits[0, -1]
                .argmax()
            )
            if value == 1:
                break
            prefix.append(value)
            generated.append(value)
        exact = (
            (tokenizer.decode(generated).strip() == str(problem["answer"]))
            if tokenizer is not None
            else generated == problem["answer_tokens"]
        )
        sums["exact"] += int(exact)
        sums["problems"] += 1
        rows.append(
            dict(
                task="arithmetic",
                group=problem["group"],
                family=problem["family"],
                carry=problem["carry"],
                answer_nll=float(answer.mean()),
                exact_match=exact,
            )
        )
    if fingerprint(model.state_dict()) != before:
        raise RuntimeError("READONLY_EVALUATION_MUTATED_MODEL")
    work = {k: v - before_work.get(k, 0) for k, v in model.work.items()}
    model.work = before_work
    return dict(
        teacher_forced_cell_usage=usage,
        story_nll=sums["story_loss"] / sums["story_tokens"],
        answer_nll=sums["answer_loss"] / sums["answer_tokens"],
        full_arithmetic_nll=sums["full_loss"] / sums["full_tokens"],
        exact_match=sums["exact"] / sums["problems"],
        rows=rows,
        seconds=time.perf_counter() - begin,
        role=role,
        scope="FIXED_32_GROUPS_BALANCED_CARRY_TWO_STORY_WINDOWS",
        counts=sums,
        operator_work=work,
    )


def repeated(model, research, role, tokenizer=None):
    a, b = measure(model, research, role, tokenizer), measure(model, research, role, tokenizer)
    error = max(abs(a[k] - b[k]) for k in ("story_nll", "answer_nll", "exact_match"))
    if error > 1e-6:
        raise RuntimeError("EVALUATION_NOT_REPRODUCIBLE")
    a["repeat_error"] = error
    return a


def attribution(old, new, research, role, tokenizer=None):
    grid = {}
    for cn, cell in [("O", old), ("N", new)]:
        for rn, connection in [("O", old), ("N", new)]:
            model = copy.deepcopy(cell)
            model.connection.load_state_dict(connection.connection.state_dict(), strict=True)
            grid[cn + rn] = measure(model, research, role, tokenizer)
    result = {}
    for metric in ("story_nll", "answer_nll"):
        v = {k: x[metric] for k, x in grid.items()}
        result[metric] = dict(
            values=v,
            V=v["OO"] - v["NN"],
            R=v["OO"] - v["ON"],
            C=v["ON"] - v["NN"],
            interaction=v["NN"] - v["NO"] - v["ON"] + v["OO"],
            cross_models="COUNTERFACTUAL_INTERVENTIONS_NOT_TRAINING_HISTORIES",
        )
    return result


def specialization(model, research, role, tokenizer=None):
    import numpy as np

    baseline = measure(model, research, role, tokenizer)
    usage = [
        sum(baseline["teacher_forced_cell_usage"][task][i] for task in ("story", "arithmetic"))
        for i in range(len(model.cells))
    ]
    rows = []
    for i in range(len(model.cells)):
        other = [j for j in range(len(model.cells)) if j != i]
        choices = {
            "FIXED_CYCLIC_CELL_ID": (i + 1) % len(model.cells),
            "USAGE_MATCHED": min(other, key=lambda j: (abs(usage[j] - usage[i]), j)),
            "RANDOM_CELL_ID": int(np.random.default_rng(153301 + i).choice(other)),
        }
        cache = {}
        for control, replacement in choices.items():
            if replacement not in cache:
                m = copy.deepcopy(model)
                m.cells[i].load_state_dict(model.cells[replacement].state_dict(), strict=True)
                cache[replacement] = measure(m, research, role, tokenizer)
            value = cache[replacement]
            story = (value["story_nll"] - baseline["story_nll"]) / max(baseline["story_nll"], 1e-12)
            arithmetic = (value["answer_nll"] - baseline["answer_nll"]) / max(
                baseline["answer_nll"], 1e-12
            )
            rows.append(
                dict(
                    cell=i,
                    replacement=replacement,
                    story_effect=story,
                    answer_effect=arithmetic,
                    specificity=arithmetic - story,
                    calls_unchanged=True,
                    control=control,
                    usage=usage[i],
                    replacement_usage=usage[replacement],
                    replacement_work=value["operator_work"],
                    scope="LEGAL_REPLACEMENT_POSTHOC_TASK_DIAGNOSTIC_NOT_ROUTER_INPUT",
                )
            )
    return rows
