"""Pure CPU content registry. Unknown coverage never grants release permission."""
import hashlib
import re
import sqlite3
import unicodedata


def content_hashes(text):
    normalized = ' '.join(unicodedata.normalize('NFKC', text).casefold().split())
    return tuple(hashlib.sha256(x.encode()).hexdigest() for x in
                 (text, normalized, ' '.join(re.findall(r'\w+', normalized))))


class ExclusionIndex:
    def __init__(self, path):
        self.db = sqlite3.connect(str(path))
        self.db.execute('CREATE TABLE IF NOT EXISTS exclusions '
                        '(method INTEGER, hash TEXT, source TEXT, role TEXT, raw_index INTEGER, '
                        'PRIMARY KEY(method, hash, source, role, raw_index))')
        self.db.execute('CREATE INDEX IF NOT EXISTS lookup ON exclusions(method, hash)')

    def add(self, text, source, role, raw_index):
        self.db.executemany('INSERT OR IGNORE INTO exclusions VALUES (?,?,?,?,?)',
                            [(i, h, source, role, raw_index)
                             for i, h in enumerate(content_hashes(text))])

    def matches(self, text):
        return [row for i, h in enumerate(content_hashes(text)) for row in
                self.db.execute('SELECT source,role,raw_index FROM exclusions '
                                'WHERE method=? AND hash=?', (i, h))]

    def close(self):
        self.db.commit()
        self.db.close()


def select(records, index, domain, count=512):
    """Rank content, never quality; collapse all three hash equivalence classes."""
    ranked = []
    for raw_index, text in records:
        if not text.strip() or index.matches(text):
            continue
        hashes = content_hashes(text)
        score = hashlib.sha256((domain + ':' + hashes[1]).encode()).hexdigest()
        ranked.append((score, raw_index, hashes))
    selected, seen = [], set()
    for score, raw_index, hashes in sorted(ranked):
        keys = set(enumerate(hashes))
        if seen.intersection(keys):
            continue
        seen.update(keys)
        selected.append(dict(raw_index=raw_index, rank_sha256=score, content_sha256=list(hashes)))
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError('INSUFFICIENT_DISTINCT_CANDIDATES')
    return selected


def release_allowed(coverage, tokenizer, windows, scientific_review):
    return all(x is True for x in (coverage, tokenizer, windows, scientific_review))
