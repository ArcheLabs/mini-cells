"""Frozen Tiny stream reconstruction and QA-isolated batch planning.

This module is deliberately model-agnostic. It never imports Torch or loads a
checkpoint. Training code can consume the returned fragments independently.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Fragment:
    source_microbatch: int
    source_start: int
    tokens: tuple
    target_kinds: tuple
    fact: str

    @property
    def input_tokens(self):
        return self.tokens[:-1]

    @property
    def labels(self):
        return self.tokens[1:]

    @property
    def label_kinds(self):
        return self.target_kinds[1:]


@dataclass(frozen=True)
class SourceRow:
    microbatch: int
    tokens: tuple
    token_kinds: tuple
    token_facts: tuple
    sampled_facts: tuple
    sampled_indices: tuple
    qa_spans: tuple

    @property
    def inputs(self):
        return self.tokens[:-1]

    @property
    def labels(self):
        return self.tokens[1:]

    @property
    def label_kinds(self):
        return self.token_kinds[1:]

    @property
    def label_facts(self):
        return self.token_facts[1:]


def _record_parts(record, eos_id):
    prompt = tuple(int(x) for x in record["prompt_ids"])
    answer = tuple(int(x) for x in record["answer_tokens"])
    if not prompt or not answer or tuple(record["ids"]) != prompt + answer:
        raise ValueError("TINY_RECORD_BOUNDARY_MISMATCH")
    return prompt, answer, int(eos_id)


def build_source_row(rng, records, sequence_tokens=65, eos_id=1, microbatch=0):
    """Mirror the frozen R3m-E sampler: sample QA rows, append EOS, truncate."""
    if sequence_tokens < 2 or not records:
        raise ValueError("SOURCE_ROW_SHAPE_INVALID")
    tokens, kinds, facts, selected, selected_indices, spans = [], [], [], [], [], []
    while len(tokens) < sequence_tokens:
        record_index = int(rng.integers(len(records)))
        record = records[record_index]
        prompt, answer, eos = _record_parts(record, eos_id)
        fact = str(record["group"])
        start = len(tokens)
        selected.append(fact)
        selected_indices.append(record_index)
        full_end = start + len(prompt) + len(answer) + 1
        tokens.extend(prompt + answer + (eos,))
        kinds.extend(("prompt",) * len(prompt) + ("answer",) * len(answer) + ("eos",))
        facts.extend((fact,) * (len(prompt) + len(answer) + 1))
        spans.append(
            {
                "fact": fact,
                "start": start,
                "end": len(tokens),
                "answer_tokens": len(answer),
                "answer_complete": start + len(prompt) + len(answer) <= sequence_tokens,
                "eos_complete": full_end <= sequence_tokens,
                "complete": full_end <= sequence_tokens,
            }
        )
    return SourceRow(
        microbatch=int(microbatch),
        tokens=tuple(tokens[:sequence_tokens]),
        token_kinds=tuple(kinds[:sequence_tokens]),
        token_facts=tuple(facts[:sequence_tokens]),
        sampled_facts=tuple(selected),
        sampled_indices=tuple(selected_indices),
        qa_spans=tuple(spans),
    )


def build_update(rng, records, rows=8, sequence_tokens=65, eos_id=1):
    if rows != 8 or sequence_tokens != 65:
        raise ValueError("D1_SOURCE_BUDGET_FROZEN")
    return tuple(
        build_source_row(rng, records, sequence_tokens, eos_id, microbatch=i) for i in range(rows)
    )


def target_masks(row):
    """Return masks indexed exactly like shifted y = source_tokens[1:]."""
    kinds = row.label_kinds
    answer_eos = tuple(kind in ("answer", "eos") for kind in kinds)
    return {
        "uniform": tuple(True for _ in kinds),
        "answer_eos": answer_eos,
        "answer": tuple(kind == "answer" for kind in kinds),
        "eos": tuple(kind == "eos" for kind in kinds),
        "label_kinds": kinds,
        "label_facts": row.label_facts,
    }


def split_independent(row):
    """Split the *already truncated* stream at QA starts.

    Each resulting QA is an independent model invocation. Its position indices
    restart at zero; it receives no earlier fragment, KV cache, or predecessor.
    The EOS-to-next-prompt transition is intentionally absent.
    """
    if not 2 <= len(row.tokens) <= 65:
        raise ValueError("SOURCE_ROW_LENGTH_CHANGED")
    starts = [0]
    for index in range(1, len(row.tokens)):
        if row.token_kinds[index] == "prompt" and row.token_kinds[index - 1] == "eos":
            starts.append(index)
    starts.append(len(row.tokens))
    out = []
    for lo, hi in zip(starts, starts[1:]):
        if hi - lo < 2:
            continue
        out.append(
            Fragment(
                source_microbatch=row.microbatch,
                source_start=lo,
                tokens=row.tokens[lo:hi],
                target_kinds=row.token_kinds[lo:hi],
                fact=row.token_facts[lo],
            )
        )
    return tuple(out)


def update_fragments(rows):
    """Return QA-local sequences plus source offsets for keyed routing draws."""
    return tuple(fragment for row in rows for fragment in split_independent(row))


def source_budget(rows):
    packed = sum(len(row.labels) for row in rows)
    split = sum(len(fragment.labels) for row in rows for fragment in split_independent(row))
    answer = sum(sum(target_masks(row)["answer"]) for row in rows)
    eos = sum(sum(target_masks(row)["eos"]) for row in rows)
    sampled = sum(len(row.qa_spans) for row in rows)
    return {
        "source_slots": packed,
        "packed_target_positions": packed,
        "independent_target_positions": split,
        "dropped_cross_qa_targets": packed - split,
        "answer_target_positions": answer,
        "eos_target_positions": eos,
        "answer_eos_target_positions": answer + eos,
        "independent_forward_count": sum(len(split_independent(row)) for row in rows),
        "complete_answer_instances": sum(
            int(span["answer_complete"]) for row in rows for span in row.qa_spans
        ),
        "complete_answer_plus_eos_instances": sum(
            int(span["eos_complete"]) for row in rows for span in row.qa_spans
        ),
        "sampled_qa_instances": sampled,
        "truncated_qa_instances": sum(
            int(not span["complete"]) for row in rows for span in row.qa_spans
        ),
    }
