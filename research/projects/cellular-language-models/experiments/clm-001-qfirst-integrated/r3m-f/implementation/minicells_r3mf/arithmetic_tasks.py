"""Preassigned facts and exact prompt/answer boundaries; no quality-dependent selection."""

from collections import Counter
from minicells_r3me.common import canonical


def encode(tokenizer, prompt, answer, group, template, maximum=64, **meta):
    answer = str(answer)
    text = prompt + answer
    e, p = tokenizer.encode(text), tokenizer.encode(prompt)
    if e.ids[: len(p.ids)] != p.ids:
        raise ValueError("TOKENIZER_PROMPT_PREFIX_NOT_EXACT")
    offsets = e.offsets
    positions = [i for i, (lo, hi) in enumerate(offsets) if hi > len(prompt)]
    covered = []
    for i in positions:
        lo, hi = offsets[i]
        if lo < len(prompt) or hi > len(text) or not text[lo:hi].isdigit():
            raise ValueError("ANSWER_MASK_INCLUDES_NON_DIGITS")
        covered.extend(range(lo, hi))
    if sorted(set(covered)) != list(range(len(prompt), len(text))) or positions != list(
        range(len(p.ids), len(e.ids))
    ):
        raise ValueError("ANSWER_MASK_NOT_EXACT")
    if not p.ids or not positions or len(e.ids) + 1 > maximum or 1 in e.ids:
        raise ValueError("TASK_SEQUENCE_OR_EOS_INVALID")
    if tokenizer.decode(e.ids[len(p.ids) :]).strip() != answer:
        raise ValueError("ANSWER_TOKEN_ROUNDTRIP_FAILED")
    return dict(
        group=group,
        template=template,
        prompt=prompt,
        text=text,
        answer=answer,
        ids=e.ids,
        prompt_ids=p.ids,
        answer_positions=positions,
        answer_tokens=[e.ids[i] for i in positions],
        sample_sha256=canonical(text),
        **meta,
    )


def carry_class(a, b):
    units = a % 10 + b % 10 >= 10
    tens = a // 10 + b // 10 + int(units) >= 10
    return "BOTH" if units and tens else "UNITS" if units else "TENS" if tens else "NONE"


def generate(c, tokenizer):
    dev = {tuple(x) for x in c["arithmetic"]["dev_facts"]}
    tiny = {tuple(x) for x in c["arithmetic"]["tiny_facts"]}
    facts = {(a, b) for a in range(10) for b in range(a, 10)}
    if len(dev) != 11 or len(tiny) != 8 or not tiny <= facts - dev:
        raise ValueError("PREASSIGNED_FACTS_INVALID")
    out = {
        "a1": {"train": [], "dev": [], "train_eval": []},
        "copy": {"train": [], "dev": []},
        "tiny": [],
    }
    for a, b in sorted(facts):
        role = "dev" if (a, b) in dev else "train"
        templates = c["arithmetic"][role + "_templates"]
        for template, text in enumerate(templates):
            # Equal weighting of fact and template: equal operands receive two identical orders.
            for order, (x, y) in enumerate(((a, b), (b, a))):
                row = encode(
                    tokenizer,
                    text.format(a=x, b=y),
                    a + b,
                    f"{a}+{b}",
                    template,
                    c["model"]["sequence_length"],
                    operands=[x, y],
                    order=order,
                )
                out["a1"][role].append(row)
                if role == "train" and template == 0 and order == 0 and (a, b) in tiny:
                    out["tiny"].append(row)
        if role == "train":
            for template, text in enumerate(c["arithmetic"]["dev_templates"]):
                for order, (x, y) in enumerate(((a, b), (b, a))):
                    out["a1"]["train_eval"].append(
                        encode(
                            tokenizer,
                            text.format(a=x, b=y),
                            a + b,
                            f"{a}+{b}",
                            template,
                            c["model"]["sequence_length"],
                            operands=[x, y],
                            order=order,
                        )
                    )
    frequency = Counter(a + b for a, b in facts - dev)
    for n in range(19):
        role = "dev" if n in c["copy"]["dev_numbers"] else "train"
        for template, text in enumerate(c["copy"][role + "_templates"]):
            row = encode(
                tokenizer, text.format(n=n), n, str(n), template, c["model"]["sequence_length"]
            )
            row["sampling_weight"] = max(1, frequency[n])
            out["copy"][role].append(row)
    return out
