"""Index and exact-content isolation; excludes duplicates without replacement."""

import hashlib

REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
TOKENIZER = "612c2aaa09e7f35c89c41c1e6270bdca1ef861a283a4de859cd808c8e6bfe6ff"
RANGES = dict(predictor_probe=(8192, 8448), value_holdout=(8448, 8960))


def boundary_proof(revision, tokenizer_hash, historical_ranges, historical_hashes, stories):
    if revision != REVISION or tokenizer_hash != TOKENIZER:
        raise ValueError("RESEARCH_DATA_IDENTITY_CHANGED")
    for start, end in RANGES.values():
        if end > 60000 or any(start < b and a < end for a, b in historical_ranges.values()):
            raise ValueError("RESEARCH_HOLDOUT_BOUNDARY_CONFLICT")
    seen = set(historical_hashes)
    result = {}
    for role, (start, end) in RANGES.items():
        accepted, excluded = [], []
        for index in range(start, end):
            if index not in stories:
                raise ValueError("STORY_RANGE_INCOMPLETE")
            digest = hashlib.sha256(stories[index].encode("utf-8")).hexdigest()
            row = dict(story_index=index, text_sha256=digest)
            if digest in seen:
                excluded.append(row)
            else:
                accepted.append(row)
                seen.add(digest)
        if not accepted:
            raise ValueError("EMPTY_RESEARCH_ROLE_AFTER_EXCLUSIONS")
        result[role] = dict(
            requested_range=[start, end],
            accepted=accepted,
            excluded=excluded,
            rule="EXCLUDE_EXACT_UTF8_SHA256_DUPLICATES_NO_REPLACEMENT",
        )
    return dict(
        revision=revision,
        tokenizer_sha256=tokenizer_hash,
        roles=result,
        confirmation="SEALED",
        historical_content_hash_count=len(set(historical_hashes)),
    )
