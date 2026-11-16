"""Paired synthetic language episodes, independent of model placement."""
import hashlib
import json
import random
from dataclasses import asdict, dataclass

SEEDS = {"train": 42, "validation": 1042, "test": 2042}
QUESTIONS = {
    "train": "Where does {} work?",
    "validation": "Who employs {}?",
    "test": "Which organization is {} employed by?",
}


@dataclass(frozen=True)
class Episode:
    facts: tuple
    distractors: tuple
    queries: tuple
    donor: tuple


def generate(split, count, associations=8):
    if associations not in (8, 16) or count < 1:
        raise ValueError("Expected 8 or 16 associations and a positive episode count")
    rng = random.Random(SEEDS[split])
    episodes = []
    for index in range(count):
        def rows(group, size):
            permutation = rng.sample(range(size), size)
            return tuple(
                (f"{split}{group}{index}e{i}", f"{split}{group}{index}v{j}")
                for i, j in enumerate(permutation)
            )

        facts = rows("F", associations)
        donor = rows("D", associations)
        distractors = rows("X", 4)
        queries = tuple(
            (QUESTIONS[split].format(entity), value, entity)
            for entity, value in rng.sample(facts, 3)
        ) + ((QUESTIONS[split].format(f"{split}{index}absent"),
              "unknown", f"{split}{index}absent"),)
        episodes.append(Episode(facts, distractors, queries, donor))
    return episodes


def digest(episodes):
    payload = json.dumps([asdict(episode) for episode in episodes],
                         sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def encode(episode, tokenizer, *, donor=False):
    """Return segment token IDs and aligned labels; -100 means no supervision."""
    def tokens(text):
        return tokenizer.encode(text, add_special_tokens=False)

    history = episode.donor if donor else episode.facts
    segments = []
    for rows in (history, episode.distractors):
        ids = tokens(" ".join(f"{entity} works for {value}." for entity, value in rows))
        segments.append((ids, [-100] * len(ids)))
    ids, labels = [], []
    for prompt, answer, _ in episode.queries:
        context = tokens(f"\nQuestion: {prompt}\nAnswer:")
        target = tokens(f" {answer}")
        if not context or not target:
            raise ValueError("Query and answer must each encode to at least one token")
        ids.extend(context + target)
        labels.extend([-100] * len(context) + target)
    segments.append((ids, labels))
    if any(not 0 < len(ids) <= 128 for ids, _ in segments):
        raise ValueError("Episode exceeds the frozen segment budget")
    return segments
