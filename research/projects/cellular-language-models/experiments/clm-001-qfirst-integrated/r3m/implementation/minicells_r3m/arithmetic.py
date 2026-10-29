"""Canonical two-digit addition; operands, answers, expressions and template families disjoint."""

from collections import Counter
from minicells_r3e.common import canonical

TEMPLATES = {
    "train": ["Calculate {a} + {b}. Answer: ", "The sum of {a} and {b} is "],
    "trigger": ["Add the numbers {a} and {b}. Result: ", "What is {a} plus {b}? Result: "],
    "pilot_report": ["Find the total of {a} with {b}. Total: ", "Evaluate {a}+{b}. Solution: "],
    "main_report": [
        "Combine {a} and {b} by addition. Value: ",
        "Compute the addition {a} + {b}. Value: ",
    ],
}


def generate(role, tokenizer, maximum_length=64):
    if role not in TEMPLATES:
        raise RuntimeError("ARITHMETIC_ROLE_FORBIDDEN")
    residue = list(TEMPLATES).index(role)
    rows, lengths = [], Counter()
    for a in range(10, 100):
        for b in range(a, 100):
            if a % 4 != residue or b % 4 != residue or (a + b) % 8 != 2 * residue:
                continue
            for family, template in enumerate(TEMPLATES[role]):
                prompt, answer = template.format(a=a, b=b), str(a + b)
                encoded = tokenizer.encode(prompt + answer)
                # Token offsets guarantee correct masking when a token spans the prompt boundary.
                answer_positions = [
                    i for i, (lo, hi) in enumerate(encoded.offsets) if hi > len(prompt)
                ]
                if (
                    not answer_positions
                    or answer_positions[0] == 0
                    or len(encoded.ids) > maximum_length + 1
                ):
                    raise RuntimeError("ARITHMETIC_TOKENIZATION_INVALID")
                target_tokens = [encoded.ids[i] for i in answer_positions]
                rows.append(
                    dict(
                        a=a,
                        b=b,
                        answer=a + b,
                        carry=(a % 10 + b % 10) >= 10,
                        family=role + "-" + str(family),
                        group=str(a) + "+" + str(b),
                        prompt=prompt,
                        text=prompt + answer,
                        ids=encoded.ids,
                        answer_positions=answer_positions,
                        answer_tokens=target_tokens,
                        prompt_ids=encoded.ids[: answer_positions[0]],
                    )
                )
                lengths[len(target_tokens)] += 1
    return rows, dict(
        raw_count=90 * 91 // 2 * 2,
        valid_count=len(rows),
        excluded_count=90 * 91 // 2 * 2 - len(rows),
        content_sha256=canonical(rows),
        answer_token_lengths=dict(lengths),
        grouping="OPERAND_MOD4_SUM_MOD8_TEMPLATE_FAMILY",
        role=role,
    )


def isolation(roles):
    fields = ("operands", "answers", "groups", "families")
    seen = {k: set() for k in fields}
    for role, rows in roles.items():
        current = dict(
            operands={v for r in rows for v in (r["a"], r["b"])},
            answers={r["answer"] for r in rows},
            groups={r["group"] for r in rows},
            families={r["family"] for r in rows},
        )
        for k in fields:
            if current[k] & seen[k]:
                raise RuntimeError("ARITHMETIC_GROUP_LEAKAGE:" + k)
            seen[k].update(current[k])
    return dict(
        status="GROUP_DISJOINT", dimensions=list(fields), roles=list(roles), confirmation="SEALED"
    )
