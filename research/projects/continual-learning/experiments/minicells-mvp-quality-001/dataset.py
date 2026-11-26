"""Frozen synthetic registry facts and disjoint question splits."""
from random import Random

VERSION = "synthetic-relays-v1"
DOMAINS = ("glassworks", "railway", "observatory", "archive",
           "greenhouse", "lighthouse", "foundry", "harbor")
COUNTS = {"smoke": (0, 3, 3), "development": (5, 8, 8), "held_out": (10, 24, 24)}
SEED = 57219


def build():
    rng = Random(SEED)
    pool = iter(rng.sample(range(10000, 99999), 4000))
    records = {domain: [] for domain in DOMAINS}
    sets = {}
    j = 0
    def add(domain, text):
        key = f"{domain}-{len(records[domain]):03d}"
        records[domain].append({"record_id": key, "text": text})
        return key

    for split, (singles, crosses, negatives) in COUNTS.items():
        samples = []
        def sample(kind, question, gold, support, references):
            nonlocal j
            j += 1
            # A fixed rotation distributes the two irrelevant controls across domains.
            domains = DOMAINS[j % 8:] + DOMAINS[:j % 8]
            samples.append(dict(sample_id=f"q{j:04}", split=split, task_type=kind,
                                prompt=question + " Reply with only the number.",
                                gold_answer=str(gold), relevant_cell_ids=support,
                                irrelevant_cell_ids=[d for d in domains if d not in support][:2],
                                answer_constraints={"type": "integer"},
                                metric_type="numeric_exact_match", source_record_ids=references,
                                dataset_version=VERSION))

        single_domains = DOMAINS[:3] if split == "smoke" else DOMAINS
        for domain in single_domains:
            for _ in range(singles or 1):
                asset, access = next(pool), next(pool)
                ref = add(domain, f"{domain.title()} inventory: item {asset} has access number {access}.")
                prompt = (f"Give the access number for inventory item {asset} in the {domain}."
                          if split == "held_out" else f"In the {domain}, which access number opens item {asset}?")
                sample("single_cell", prompt, access, [domain], [ref])

        for i in range(crosses):
            left, right = DOMAINS[i % 8], DOMAINS[(i + 1 + i // 8) % 8]
            asset, relay, value = next(pool), next(pool), next(pool)
            a = add(left, f"{left.title()} dispatch: item {asset} is assigned relay {relay}.")
            b = add(right, f"{right.title()} lookup: relay {relay} resolves to destination number {value}.")
            prompt = (f"Find the destination number for {left} item {asset} using its assigned relay "
                      f"and the {right} lookup."
                      if split == "held_out" else
                      f"Using the {right} relay lookup, what destination number serves {left} item {asset}?")
            sample("cross_cell", prompt, value, [left, right], [a, b])

        for i in range(negatives):
            left, right = rng.randint(12, 99), rng.randint(12, 99)
            prompt = (f"A crate has {left} blue tiles and {right} red tiles. How many tiles are in it?"
                      if split == "held_out" else f"Compute the sum of {left} and {right}.")
            sample("irrelevant", prompt, left + right, [], [])
        sets[split] = samples
    return records, sets
