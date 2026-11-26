"""Fixed public-knowledge continuations for a pretrained Base model."""

FIXTURES = (
    ("france", "The capital of France is", "Paris", "What is the capital of France?"),
    ("days", "Monday, Tuesday, Wednesday, Thursday,", "Friday", "What day comes after Thursday?"),
    ("numbers", "One, two, three, four,", "five", "What word comes after four in counting?"),
    ("color", "A ripe banana is usually", "yellow", "What color is a ripe banana usually?"),
    ("arithmetic", "2 + 2 =", "4", "What is two plus two?"),
    ("opposite", "The opposite of hot is", "cold", "What is the opposite of hot?"),
)


def prompts():
    for key, prefix, target, question in FIXTURES:
        for style, text in (
            ("continuation", prefix),
            ("fact_completion", f"{prefix} {target}.\n{prefix}"),
            ("question_answer", f"Question: {question} Reply with only the answer.\nAnswer:"),
        ):
            yield dict(id=key, style=style, text=text, target=target)
