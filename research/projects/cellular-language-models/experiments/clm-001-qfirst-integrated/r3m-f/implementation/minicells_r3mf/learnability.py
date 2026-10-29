"""Free generation and fact-level capability; no teacher-forced proxy for exact match."""

import copy
import math
import time
from collections import defaultdict
from contextlib import contextmanager
import numpy as np
import torch
from torch.nn import functional as F
from minicells_r3me.evaluation import _path_keys
from minicells_r3me.common import fingerprint


@contextmanager
def preserve(model):
    before = fingerprint(model.state_dict())
    saved = (
        copy.deepcopy(model.work),
        model.training,
        model.training_context,
        model.observer,
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )
    try:
        model.eval()
        model.training_context = None
        model.observer = None
        yield
    finally:
        model.work = saved[0]
        model.train(saved[1])
        model.training_context = saved[2]
        model.observer = saved[3]
        torch.set_rng_state(saved[4])
        if saved[5] is not None:
            torch.cuda.set_rng_state_all(saved[5])
        if fingerprint(model.state_dict()) != before:
            raise RuntimeError("EVALUATION_MUTATED_MODEL")


@torch.no_grad()
def generate(model, prompt, cap, eos=1):
    prefix = list(prompt)
    result = []
    ended = False
    device = next(model.parameters()).device
    for _ in range(cap):
        x = torch.tensor([prefix[-model.c["model"]["max_sequence_length"] :]], device=device)
        token = int(model(x).logits[0, -1].argmax())
        if token == eos:
            ended = True
            break
        result.append(token)
        prefix.append(token)
    return result, ended


def wilson(success, n):
    if not n:
        return None
    p = success / n
    z = 1.959963984540054
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0, mid - half), min(1, mid + half)]


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    facts = []
    for group, part in sorted(grouped.items()):
        facts.append(
            dict(
                group=group,
                answer=part[0]["answer"],
                exact_match=sum(float(x["exact_match"]) for x in part) / len(part)
                if part[0]["exact_match"] is not None
                else None,
                first_token_accuracy=sum(float(x["first_token_correct"]) for x in part) / len(part),
                answer_nll=sum(x["answer_nll"] for x in part) / len(part),
                full_nll=sum(x["full_nll"] for x in part) / len(part),
            )
        )

    def average(key):
        return (
            sum(x[key] for x in facts) / len(facts) if facts and facts[0][key] is not None else None
        )

    em = average("exact_match")
    # A fact success for the binomial interval means ALL frozen variants are correct.
    strict = sum(x["exact_match"] == 1 for x in facts) if em is not None else None
    strata = {}
    for name, pred in (
        ("one_digit", lambda r: int(r["answer"]) < 10),
        ("two_digit", lambda r: int(r["answer"]) >= 10),
    ):
        part = [x for x in facts if pred(x)]
        strata[name] = dict(
            facts=len(part),
            exact_match=sum(x["exact_match"] for x in part) / len(part)
            if part and em is not None
            else None,
        )
    templates = {}
    for t in sorted({x["template"] for x in rows}):
        part = [x for x in rows if x["template"] == t]
        per = defaultdict(list)
        for r in part:
            per[r["group"]].append(r)
        templates[str(t)] = {
            "facts": len(per),
            "exact_match": sum(
                sum(float(r["exact_match"]) for r in rs) / len(rs) for rs in per.values()
            )
            / len(per)
            if em is not None
            else None,
        }
    return dict(
        facts=len(facts),
        fact_rows=facts,
        answer_nll=average("answer_nll"),
        full_nll=average("full_nll"),
        exact_match=em,
        first_token_accuracy=average("first_token_accuracy"),
        strict_all_variants_successes=strict,
        strict_fact_wilson95=wilson(strict, len(facts)) if strict is not None else None,
        interval_scope="ALL_VARIANTS_CORRECT_FACT_INDICATOR; descriptive variation-averaged EM is separate",
        strata=strata,
        templates=templates,
        predictions=rows,
    )


def _exposure(out, cells):
    paths = _path_keys(out)
    stage = np.zeros((8, cells), dtype=np.int64)
    for path in paths:
        if len(path) != 8:
            raise ValueError("HARD_PATH_DEPTH_CHANGED")
        for k, a in enumerate(path):
            stage[k, int(a)] += 1
    return stage


@torch.no_grad()
def task_metrics(model, records, tokenizer, c, decode=True):
    device = next(model.parameters()).device
    rows = []
    stage = np.zeros((8, len(model.cells)), dtype=np.int64)
    for row in records:
        ids = row["ids"] + [c["arithmetic"]["eos_token_id"]]
        out = model(torch.tensor([ids[:-1]], device=device))
        stage += _exposure(out, len(model.cells))
        losses = F.cross_entropy(
            out.logits.reshape(-1, out.logits.shape[-1]),
            torch.tensor(ids[1:], device=device),
            reduction="none",
        )
        positions = [i - 1 for i in row["answer_positions"]]
        answer = losses[positions]
        first = bool(int(out.logits[0, positions[0]].argmax()) == row["answer_tokens"][0])
        tokens, ended = (
            generate(
                model,
                row["prompt_ids"],
                c["evaluation"]["autoregressive_answer_token_cap"],
                c["arithmetic"]["eos_token_id"],
            )
            if decode
            else ([], False)
        )
        prediction = tokenizer.decode(tokens, skip_special_tokens=False).strip() if decode else None
        if decode:
            first = bool(tokens and tokens[0] == row["answer_tokens"][0])
        rows.append(
            dict(
                group=row["group"],
                template=row["template"],
                order=row.get("order", 0),
                answer=row["answer"],
                prediction=prediction,
                generated_ids=tokens,
                eos_reached=ended,
                exact_match=(prediction == row["answer"] and ended) if decode else None,
                text_only_match=prediction == row["answer"] if decode else None,
                first_token_correct=first,
                first_token_source="FREE_GENERATION" if decode else "CAUSAL_PREFIX_GREEDY",
                answer_nll=float(answer.mean()),
                full_nll=float(losses.mean()),
                eos_nll=float(losses[-1]),
            )
        )
    return aggregate(rows), stage.tolist()


@torch.no_grad()
def measure(model, research, tokenizer, c, decode=True, tiny=False):
    start = time.perf_counter()
    with preserve(model):
        if tiny:
            metric, exposure = task_metrics(model, research["tiny"], tokenizer, c, decode)
            return dict(tiny=metric, exposure={"a1": exposure}, seconds=time.perf_counter() - start)
        device = next(model.parameters()).device
        losses = []
        stage = np.zeros((8, len(model.cells)), dtype=np.int64)
        story_rows = []
        for group in research["story"]["development"]:
            part = []
            for ids in group["windows"]:
                out = model(torch.tensor([ids[:-1]], device=device))
                part.append(
                    float(
                        F.cross_entropy(
                            out.logits.reshape(-1, out.logits.shape[-1]),
                            torch.tensor(ids[1:], device=device),
                        )
                    )
                )
                stage += _exposure(out, len(model.cells))
            story_rows.append(dict(group=group["group"], nll=sum(part) / len(part)))
            losses.extend(part)
        result = dict(
            story_nll=sum(losses) / len(losses),
            story_rows=story_rows,
            exposure={"story": stage.tolist()},
        )
        for task, role in (("copy", "dev"), ("a1", "dev"), ("a1_train", "train_eval")):
            source = "a1" if task == "a1_train" else task
            result[task], exposure = task_metrics(
                model, research[source][role], tokenizer, c, decode
            )
            if task != "a1_train":
                result["exposure"][task] = exposure
        result["exposure_summary"] = {}
        for task, counts in result["exposure"].items():
            array = np.asarray(counts)
            total = array.sum(0)
            result["exposure_summary"][task] = dict(
                cell_counts=total.tolist(),
                cell_frequency=(total / max(int(total.sum()), 1)).tolist(),
                stage_frequency=(array / np.maximum(array.sum(1, keepdims=True), 1)).tolist(),
                source="DETERMINISTIC_TEACHER_FORCED_PATHS_NOT_CAPABILITY",
            )
        result["seconds"] = time.perf_counter() - start
        return result
