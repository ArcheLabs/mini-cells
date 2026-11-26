"""Answer-only likelihood and SDK-native deterministic generation."""
from collections import Counter
from contextlib import nullcontext
import math
import re
import time

import torch

from minicells_native.multicell_fusion import FusionReader
from minicells_native.query import resolve_selection, run_query, working_step


class Trace:
    stop_on_eos = True

    def __init__(self, backbone):
        self.eos = backbone.tokenizer.eos_token_id
        self.tokens = []
        self.reason = "max_new_tokens"
        self.steps = 0

    def phase(self, *_):
        return nullcontext()

    def logits(self, logits, token):
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite generation logits")
        self.steps += 1
        if token == self.eos:
            self.reason = "eos"

    def cache(self, _):
        pass


def prompt_ids(tokenizer, prompt):
    return tokenizer.encode(f"Question: {prompt}\nAnswer:", add_special_tokens=False)


def answer_ids(tokenizer, prompt, gold):
    if not gold.strip():
        raise ValueError("Empty answer")
    context = prompt_ids(tokenizer, prompt)
    whole = tokenizer.encode(f"Question: {prompt}\nAnswer: {gold}", add_special_tokens=False)
    if not context or whole[:len(context)] != context or len(whole) <= len(context):
        raise ValueError("Answer does not start on a stable token boundary")
    target = whole[len(context):]
    if any(i in tokenizer.all_special_ids for i in target):
        raise ValueError("Unexpected special token in target")
    return target


@torch.inference_mode()
def nll(backbone, step, context, target):
    """Feed the prompt in SDK chunks and each prior gold token one at a time.

    Inputs are unpadded, batch size is one, and EOS is excluded from targets.
    """
    if not context or not target or len(context) + len(target) > 2048:
        raise ValueError("Empty or truncated teacher-forced target")
    for start in range(0, len(context), 64):
        hidden = step(context[start:start + 64])
    values = []
    for index, token in enumerate(target):
        logits = backbone.model.lm_head(hidden[:, -1]).float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite teacher-forced logits")
        values.append(float(-logits.log_softmax(-1)[0, token]))
        if index + 1 < len(target):
            hidden = step([token])
    mean = sum(values) / len(values)
    return dict(target_token_count=len(values), target_token_ids=target, target_token_nll=values,
                mean_target_nll=mean, target_perplexity=math.exp(mean))


def metrics(text, gold):
    answer = text.strip()
    valid = re.fullmatch(r"[+-]?\d{1,12}", answer, flags=re.ASCII) is not None
    matches = bool(valid and int(answer) == int(gold))
    picked, expected = Counter(re.findall(r"\w+", answer.lower())), Counter([gold])
    overlap = sum((picked & expected).values())
    f1 = 2 * overlap / (sum(picked.values()) + 1)
    return dict(exact_match=matches, token_f1=f1, task_accuracy=matches,
                constraint_satisfied=valid, parse_failure=not valid)


@torch.inference_mode()
def question_only(runtime, prompt, requested, max_new, do_generate):
    """The selector receives prompt and Cell IDs, with no target or sample row."""
    backbone = runtime._backbone
    context = prompt_ids(backbone.tokenizer, prompt)
    if not context or len(context) + max_new > 2048:
        raise ValueError("Empty or truncated prompt")
    with runtime._operation():
        if do_generate:
            observer = Trace(backbone)
            result = run_query(runtime, prompt, requested, max_new, 0, token_ids=context, probe=observer)
            selected = list(result.selected_cell_ids)
            trace = dict(candidates=result.candidate_cell_ids, scores=result.cell_scores,
                         ranked=result.diagnostics["ranked_cell_ids"], selected=selected,
                         executed=observer.executed, unscored=result.diagnostics["unscored_cell_ids"],
                         fallback=result.fallback_reason, selection_seconds=result.diagnostics["selection_seconds"],
                         scoring_seconds=result.diagnostics["scoring_seconds"],
                         fusion_mode=result.diagnostics["fusion_mode"])
            generated = dict(text=result.text, token_ids=observer.tokens, stop_reason=observer.reason,
                             generated_token_count=result.generation_tokens, latency_seconds=result.latency)
        else:
            mode, candidates = runtime._candidate_ids(requested)
            selected, scores, ranked, fallback, _, scoring, choice = resolve_selection(
                runtime, FusionReader(backbone), context, mode, candidates)
            trace = dict(candidates=candidates, scores=scores, ranked=ranked, selected=selected,
                         executed=None, fallback=fallback, unscored=[c for c in candidates if c not in scores]
                         if mode == "auto" else [], selection_seconds=choice, scoring_seconds=scoring,
                         fusion_mode="hierarchical_equal" if selected else "native")
            generated = None
    return context, trace, generated


@torch.inference_mode()
def teacher(runtime, context, target, selected):
    begin = time.monotonic()
    with runtime._operation():
        step, _ = working_step(runtime, FusionReader(runtime._backbone), selected)
        result = nll(runtime._backbone, step, context, target)
    return {**result, "latency_seconds": time.monotonic() - begin}
