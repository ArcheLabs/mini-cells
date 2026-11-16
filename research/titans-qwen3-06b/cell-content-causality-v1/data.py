"""Deterministic paired histories for content intervention."""
import hashlib
import json
import random

SPLITS = {"train": (42, 100), "validation": (1042, 32), "test": (2042, 32)}
TEMPLATES = ("Where does {} work?", "Which company employs {}?")


def episodes(split):
    seed, count = SPLITS[split]
    rng = random.Random(seed)
    for index in range(count):
        numbers = rng.sample(range(10000), 28)
        names = [f"Person {seed * 1000000 + index * 10000 + n}" for n in numbers[:14]]
        values = [f"Company {seed * 1000000 + index * 10000 + n}" for n in numbers[14:]]
        facts = list(zip(names[:8], values[:8]))
        distractors = list(zip(names[8:12], values[8:12]))
        slots = rng.sample(range(8), 4)
        queries = []
        for slot in slots:
            alternative = values[12 + len(queries) % 2]
            changed = facts.copy()
            changed[slot] = (names[slot], alternative)
            queries.append({"entity": names[slot], "answer": values[slot],
                            "counterfactual": alternative, "counterfactual_facts": changed})
        queries.extend({"entity": name, "answer": "unknown"} for name in names[12:])
        if split == "train":
            queries = [rng.choice(queries[:4] if index % 2 == 0 else queries[4:])]
        for i, query in enumerate(queries):
            template = 0 if split == "train" else i % len(TEMPLATES)
            query["prompt"] = TEMPLATES[template].format(query["entity"])
            query["template"] = template
        yield {"facts": facts, "distractors": distractors, "queries": queries}


def manifest(split):
    rows = list(episodes(split))
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {"split": split, "seed": SPLITS[split][0], "episodes": len(rows),
            "sha256": hashlib.sha256(payload).hexdigest()}
