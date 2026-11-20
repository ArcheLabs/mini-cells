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
    entailment = next(int(i) for i,label in model.config.id2label.items()
                      if label.casefold()=='entailment')
    from run import CLAIMS
    fixture=json.loads((ROOT/'fixture.json').read_text())
    events={e[0]:e[2] for e in fixture['events']}
    cases={q[0]:[CLAIMS[f] for f in q[2]] for q in fixture['queries']}
    traces={r['id']:r for r in json.loads((ROOT/'results.json').read_text())['rows']}
    result['candidate_source_sha256']=hashlib.sha256((ROOT/'results.json').read_bytes()).hexdigest()
    rows = []
    with torch.inference_mode():
        for qid,claims in cases.items():
            candidates=traces[qid]['candidates'][:16]
            for claim in claims:
                inputs=tokenizer([events[c['event']] for c in candidates],
                                 [claim]*len(candidates),padding=True,truncation=True,
                                 max_length=384,return_tensors='pt')
                logits=model(**inputs).logits
                probabilities=logits.softmax(-1).tolist()
                scores=[dict(event=c['event'],cell=c['cell'],grade=c['grade'],dense=c['dense'],
                             entailment=score[entailment], probabilities={model.config.id2label[j]:v for j,v in enumerate(score)}) for c,score in zip(candidates,probabilities)]
                rows.append(dict(query_id=qid,claim=claim,
                                 candidates=sorted(scores,key=lambda c:-c['entailment'])))
    result.update(status='SCORED',torch=str(torch.__version__),transformers=transformers.__version__,
                  label_mapping=model.config.id2label,rows=rows)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    return 0


if __name__=='__main__':
    sys.exit(main())
