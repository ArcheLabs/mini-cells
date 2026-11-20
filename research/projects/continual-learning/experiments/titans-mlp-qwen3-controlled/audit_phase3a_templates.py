"""Review rendered template spans using the pinned local tokenizer."""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
from audit_qwen import sha
from phase3a_templates import TEMPLATES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    args = parser.parse_args()
    expected = json.loads(Path(__file__).with_name('qwen-pretrained-audit.json').read_text())
    if sha(args.snapshot/'tokenizer.json') != expected['hashes']['tokenizer.json']:
        raise ValueError('Tokenizer fingerprint mismatch')
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    rows = []
    for template in TEMPLATES.values():
        text, spans = template.render('E123456', 'V543210')
        tokens = tokenizer(text, return_offsets_mapping=True)
        indices = []
        for start, end in spans:
            hits = [i for i, (a,b) in enumerate(tokens.offset_mapping) if a < end and b > start]
            if not hits or (indices and hits[0] <= indices[-1][-1]):
                raise ValueError(f'Invalid spans in {template.template_id}')
            indices.append(hits)
        if template.role == 'write' and tokens.offset_mapping[indices[1][-1]][1] > spans[2][0]:
            raise ValueError('Relation token crosses into the value')
        rows.append(dict(id=template.template_id, relation=template.relation, role=template.role,
                         family=template.family, template=template.text, rendered=text,
                         character_spans=spans, token_spans=indices))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
