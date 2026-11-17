"""Reconstruct WikiText-103 articles from the pinned Stage 6B corpus."""
import json
from pathlib import Path
import re
import sys

from datasets import load_dataset
from transformers import AutoTokenizer

from train import MODEL, REVISION

SOURCE = "Salesforce/wikitext"
CORPUS = "b08601e04326c79dfdd32d625aee71d232d685c3"
TITLE = re.compile(r"^=\s+[^=]+?\s+=$")


def articles(rows):
    pending = []
    for row in rows:
        text = row["text"].strip()
        if TITLE.fullmatch(text):
            if pending:
                yield "\n".join(pending)
            pending = [text]
        elif pending and text:
            pending.append(text)
    if pending:
        yield "\n".join(pending)


def prepare(path):
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    seen = set()
    with path.open("w") as output:
        for split, quota in (("train", 200), ("validation", 8)):
            count = 0
            rows = load_dataset(SOURCE, "wikitext-103-raw-v1", revision=CORPUS,
                                split=split, streaming=True)
            for article in articles(rows):
                ids = tuple(tokenizer.encode(article, add_special_tokens=False)[:2048])
                if len(ids) != 2048 or ids in seen:
                    continue
                seen.add(ids)
                output.write(json.dumps({"text": article, "split": split}) + "\n")
                count += 1
                if count == quota:
                    break
            if count != quota:
                raise ValueError(f"{split} has {count}/{quota} qualifying articles")
            print({"split": split, "documents": count}, flush=True)


if __name__ == "__main__":
    prepare(Path(sys.argv[1]))
