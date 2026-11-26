"""Independent Cell scoring followed by a fresh selective generation pass."""
from contextlib import nullcontext
import math
import time

import torch

from .generation import generate
from .multicell_fusion import FusionConfig, FusionReader
from .sdk import QueryResult

SELECTIVE = FusionConfig("hierarchical_equal")


class SharedNorm:
    timing = True
    removed_unit = None

    def __init__(self):
        self.values = []

    def measure(self, *_):
        return nullcontext()

    def shared_norm(self, output):
        # Average L2 norms over head/token/layer as in the residual experiment.
        self.values.append(output.float().norm(dim=-1).mean())

    def score(self):
        value = float(torch.stack(self.values).mean())
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite Shared Read Output Norm")
        return value


@torch.inference_mode()
def score_cells(runtime, reader, tokens, candidates, probe=None):
    phase = probe.phase if probe else lambda *_: nullcontext()
    scores = {}
    for cell in candidates:
        if not runtime.cells._get(cell)["tokens_written"]:
            continue
        trace = SharedNorm()
        with phase("cell_load", cell):
            caches = reader.start([runtime.cells._state(cell)])
        total = 0.0
        for start in range(0, len(tokens), 64):
            chunk = tokens[start:start + 64]
            trace.values.clear()
            with phase("independent_read", cell):
                reader.step(chunk, caches, SELECTIVE, diagnostics=trace)
            with phase("norm_reduce", cell):
                total += len(chunk) * trace.score()
        scores[cell] = total / len(tokens)
        if probe:
            probe.cache(caches)
        del caches
    return scores


def select_cells(runtime, scores):
    ranking = sorted(scores, key=lambda cell: (-scores[cell], cell))
    selected = [cell for cell in ranking if runtime._threshold is None
                or scores[cell] >= runtime._threshold][:runtime._top_k]
    return ranking, selected


def resolve_selection(runtime, reader, tokens, mode, candidates, weights=None, probe=None):
    phase = probe.phase if probe else lambda *_: nullcontext()
    scores, fallback, ranking = {}, None, []
    scoring_started = time.perf_counter()
    if mode == "auto":
        with phase("scoring"):
            scores = score_cells(runtime, reader, tokens, candidates, probe)
    scoring = time.perf_counter() - scoring_started

    selection_started = time.perf_counter()
    if mode == "auto":
        with phase("selector"):
            ranking, selected = select_cells(runtime, scores)
        if not selected:
            fallback = "no_match" if candidates else "no_registered_cells"
    else:
        selected = list(candidates)
        if weights is not None and candidates:
            total = sum(weights)
            if (len(weights) != len(candidates) or not math.isfinite(total) or total <= 0
                    or any(not math.isfinite(weight) or weight < 0 for weight in weights)):
                raise ValueError("Invalid manual fusion weights")
            selected = [cell for cell, weight in zip(candidates, weights, strict=True) if weight > 0]
            weights = [weight / total for weight in weights if weight > 0]
        if not selected:
            fallback = "explicit_no_cells"
    selection = time.perf_counter() - selection_started

    return selected, scores, ranking, fallback, weights, scoring, selection


def working_step(runtime, reader, selected, weights=None, probe=None):
    backbone = runtime._backbone
    phase = probe.phase if probe else lambda *_: nullcontext()
    with phase("selected_cell_load"):
        caches = (reader.start(runtime.cells._state(cell) for cell in selected)
                  if selected else [backbone.stream({})])
    if probe:
        probe.executed = list(selected)
    if selected and weights is None:
        def step(ids):
            with phase("fusion_forward"):
                return reader.step(ids, caches, SELECTIVE)
    else:
        def step(ids):
            return backbone.step(ids, caches, weights if selected else [1.0])
    return step, caches


@torch.inference_mode()
def run_query(runtime, prompt, cells, max_new_tokens, temperature, weights=None, *, token_ids=None, probe=None):
    started = time.perf_counter()
    phase = probe.phase if probe else lambda *_: nullcontext()
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Query prompt must be nonempty text")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 512:
        raise ValueError("max_new_tokens must be between 1 and 512")
    if not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError("temperature must be between 0 and 2")
    mode, candidates = runtime._candidate_ids(cells)
    backbone = runtime._backbone
    tokens = (backbone.tokenizer.encode(f"Question: {prompt}\nAnswer:", add_special_tokens=False)
              if token_ids is None else token_ids)
    if not tokens or len(tokens) + max_new_tokens > 2048:
        raise ValueError("Query and generation must fit within 2048 tokens")
    reader = FusionReader(backbone)
    cuda = torch.device(backbone.device).type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(backbone.device)

    selected, scores, ranking, fallback, weights, scoring, selection = resolve_selection(
        runtime, reader, tokens, mode, candidates, weights, probe)

    generation_started = time.perf_counter()
    if probe:
        probe.selected = list(selected)
    step, caches = working_step(runtime, reader, selected, weights, probe)
    # Keep the same working caches across chunks and generation. The prefill path
    # handles queries longer than T4's recurrent limit without using Chunk kernels.
    final_start = ((len(tokens) - 1) // 64) * 64
    with phase("prefill"):
        for i in range(0, final_start, 64):
            step(tokens[i:i + 64])
        hidden = step(tokens[final_start:]) if probe else None
    tail = tokens[final_start:]
    with phase("generation"):
        if probe:
            def decode(ids):
                nonlocal hidden
                if hidden is not None:
                    result, hidden = hidden, None
                    return result
                return step(ids)
            text, answer = generate(backbone, tail, decode, max_new_tokens, temperature,
                                    stop_on_eos=probe.stop_on_eos, observe=probe.logits)
            probe.tokens = answer
            probe.cache(caches)
        else:
            text, answer = generate(backbone, tail, step, max_new_tokens, temperature)
    generation = time.perf_counter() - generation_started
    peak = torch.cuda.max_memory_allocated(backbone.device) if cuda else None
    diagnostics = dict(scoring_seconds=scoring, selection_seconds=selection,
                       fusion_and_generation_seconds=generation, peak_gpu_bytes=peak,
                       selection_top_k=runtime._top_k, selection_threshold=runtime._threshold,
                       ranked_cell_ids=ranking, fusion_mode=(SELECTIVE.mode if weights is None else "legacy_weighted") if selected else "native",
                       unscored_cell_ids=[cell for cell in candidates if cell not in scores]
                       if mode == "auto" else [])
    return QueryResult(text, mode, candidates, selected, scores, fallback,
                       time.perf_counter() - started, len(answer), diagnostics)
