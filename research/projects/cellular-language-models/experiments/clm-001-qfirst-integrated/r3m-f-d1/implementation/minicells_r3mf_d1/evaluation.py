"""Free-generation metric contract for all eight frozen Tiny facts."""


def generate_tokens(next_token, prompt_ids, max_tokens=8, eos_id=1):
    """Greedy autoregressive helper; returned text tokens exclude EOS and prompt."""
    if max_tokens < 0 or eos_id < 0:
        raise ValueError("D1_GENERATION_LIMIT_INVALID")
    prefix = list(prompt_ids)
    continuation = []
    ended = False
    for _ in range(max_tokens):
        token = int(next_token(tuple(prefix)))
        if token == eos_id:
            ended = True
            break
        continuation.append(token)
        prefix.append(token)
    return continuation, ended


def teacher_forced_nll(per_token_nll, target_kinds):
    """Separate gold-conditioned digit and EOS NLL from free-generation scores."""
    if len(per_token_nll) != len(target_kinds):
        raise ValueError("D1_EVAL_NLL_TARGET_ALIGNMENT")
    answer = [float(value) for value, kind in zip(per_token_nll, target_kinds) if kind == "answer"]
    eos = [float(value) for value, kind in zip(per_token_nll, target_kinds) if kind == "eos"]
    if not answer or not eos:
        raise ValueError("D1_EVAL_ANSWER_OR_EOS_TARGET_MISSING")
    return {
        "answer_nll": sum(answer) / len(answer),
        "eos_nll": sum(eos) / len(eos),
        "answer_target_count": len(answer),
        "eos_target_count": len(eos),
        "teacher_forced": True,
    }


def summarize_predictions(rows):
    if len(rows) != 8:
        raise ValueError("D1_EVALUATION_REQUIRES_ALL_EIGHT_FACTS")
    out = []
    for row in rows:
        expected = tuple(int(x) for x in row["expected_answer_tokens"])
        generated = tuple(int(x) for x in row["generated_answer_tokens"])
        eos = bool(row["eos_generated"])
        out.append(
            {
                "fact": row["fact"],
                "expected_answer_tokens": list(expected),
                "generated_answer_tokens": list(generated),
                "eos_generated": eos,
                "first_token_correct": bool(generated and expected and generated[0] == expected[0]),
                "text_exact_match": generated == expected,
                "strict_exact_match": generated == expected and eos,
                "answer_nll": float(row["answer_nll"]),
                "eos_nll": float(row["eos_nll"]),
            }
        )
    return {
        "predictions": out,
        "first_token_accuracy": sum(x["first_token_correct"] for x in out) / 8,
        "text_only_exact_match": sum(x["text_exact_match"] for x in out) / 8,
        "strict_exact_match": sum(x["strict_exact_match"] for x in out) / 8,
        "eos_termination_rate": sum(x["eos_generated"] for x in out) / 8,
        "answer_nll": sum(x["answer_nll"] for x in out) / 8,
        "eos_nll": sum(x["eos_nll"] for x in out) / 8,
        "exact_match_requires_eos": True,
        "text_exact_and_strict_exact_reported_separately": True,
    }
