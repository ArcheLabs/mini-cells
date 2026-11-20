"""Frozen candidate NLI probe; no fitting and no MA-002 access."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('revision')
    p.add_argument('output', type=Path)
    p.add_argument('--download', action='store_true')
    p.add_argument('--snapshot', type=Path)
    a = p.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', a.revision):
        p.error('Use an immutable model commit')
    model_id = 'cross-encoder/nli-deberta-v3-small'
    result = dict(model=model_id, revision=a.revision)
    if a.snapshot and (a.snapshot/'model.safetensors').is_file():
        with (a.snapshot/'model.safetensors').open('rb') as f:
            result['weights_sha256'] = hashlib.file_digest(f, 'sha256').hexdigest()
    try:
        import torch
        import transformers
        options = dict(revision=a.revision, local_files_only=not a.download)
        tokenizer = transformers.AutoTokenizer.from_pretrained(a.snapshot or model_id, **options)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(a.snapshot or model_id, **options).eval()
    except (ImportError, OSError) as exc:
        result.update(status='BLOCKED', reason=str(exc))
        a.output.write_text(json.dumps(result,indent=2)+'\n')
        return 2
    torch.set_num_threads(4)
    raw=(ROOT/'pairs.json').read_bytes()
    result['fixture_sha256']=hashlib.sha256(raw).hexdigest()
    rows=[]
    with torch.inference_mode():
        for name,present,absent,claim in json.loads(raw):
            inputs=tokenizer([present,absent],[claim,claim],padding=True,
                             truncation=True,max_length=384,return_tensors='pt')
            probs=model(**inputs).logits.softmax(-1).tolist()
            scores=[{model.config.id2label[j].lower():p for j,p in enumerate(row)} for row in probs]
            rows.append(dict(id=name,claim=claim,present=scores[0],absent=scores[1]))
    result.update(status='SCORED',torch=str(torch.__version__),transformers=transformers.__version__,
                  rows=rows,summary={group:dict(
                  pairs=len(selected),
                  higher_with_evidence=sum(r['present']['entailment']>r['absent']['entailment'] for r in selected),
                  unsupported_entailment_argmax=sum(max(r['absent'],key=r['absent'].get)=='entailment' for r in selected))
                  for group in ('actor','attribute','relation')
                  for selected in [[r for r in rows if r['id'].startswith(group+'-')]]})
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    return 0


if __name__=='__main__':
    sys.exit(main())
