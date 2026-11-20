"""Development-only complete judgments; optional frozen E5/NLI snapshots."""
import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLAIMS = {
    'restore': 'A fresh process imports saved fast weights after a notebook restart.',
    'verify': 'The loader verifies the snapshot checksum before accepting restored fast weights.',
    'map': 'A source map translates executable offsets to original source locations.',
    'trace': 'The validator records instruction traces when replaying work packages.',
    'symbols': 'A debug-symbol table records function names alongside executable addresses.',
    'passcode': 'A passcode is required to open the Portable Cell snapshot.',
    'approve': 'The record identifies the vendor that approved the snapshot format.',
    'boiling': 'The record states the boiling point of mercury.',
}


def tokens(s):
    return set(re.findall(r'[a-z0-9]+', s.lower())) - set(
        'a an the is are and to of in what which where how for after when do we'.split())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('output', type=Path)
    p.add_argument('--e5', type=Path)
    p.add_argument('--nli', type=Path)
    a = p.parse_args()
    fixture = json.loads((HERE/'fixture.json').read_text())
    events, queries = fixture['events'], fixture['queries']
    dense = None
    if a.e5:
        import torch
        from transformers import AutoTokenizer, AutoModel
        torch.set_num_threads(4)
        tok = AutoTokenizer.from_pretrained(a.e5, local_files_only=True)
        enc = AutoModel.from_pretrained(a.e5, local_files_only=True).eval()
        with torch.inference_mode():
            batch = tok(['query: '+q[1] for q in queries]+['passage: '+e[2] for e in events],
                        padding=True, truncation=True, return_tensors='pt')
            mask = batch.attention_mask.unsqueeze(-1)
            vec = torch.nn.functional.normalize((enc(**batch).last_hidden_state*mask).sum(1),dim=-1)
            dense = (vec[:len(queries)]@vec[len(queries):].T).tolist()
    nli = None
    if a.nli:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        nlitok = AutoTokenizer.from_pretrained(a.nli, local_files_only=True)
        nli = AutoModelForSequenceClassification.from_pretrained(a.nli, local_files_only=True).eval()
    rows = []
    for qi,(qid,query,facets,expected,grade) in enumerate(queries):
        assert len(grade)==len(events) and set(grade)<=set('0123')
        lex = [len(tokens(query)&tokens(e[2]))/max(1,len(tokens(query))) for e in events]
        # Fixed discovery, independent of labels: rank fusion (k=60).
        channels = [lex] + ([dense[qi]] if dense is not None else [])
        ranks = [{i:r for r,i in enumerate(sorted(range(len(events)),key=lambda j:(-s[j],j)),1)
                  if s[i]>0} for s in channels]
        ids = sorted(range(len(events)),key=lambda i:(-sum(1/(60+r[i]) for r in ranks if i in r),i))[:16]
        candidates = [dict(event=events[i][0],cell=events[i][1],grade=int(grade[i]),
                           lexical=lex[i],dense=dense[qi][i] if dense is not None else None,
                           structured=events[i][3] in facets) for i in ids]
        covered = {events[i][3] for i in ids}&set(facets)
        related = any(lex[i]>0 for i in ids)
        status = 'SUPPORTED' if covered==set(facets) else 'RELATED' if covered or related else 'NO_MATCH'
        if nli is not None:
            with torch.inference_mode():
                for facet in facets:
                    batch = nlitok([events[i][2] for i in ids],[CLAIMS[facet]]*len(ids),
                                   padding=True,truncation=True,return_tensors='pt')
                    for c,probs in zip(candidates,nli(**batch).logits.softmax(-1).tolist()):
                        c.setdefault('nli',{})[facet]={nli.config.id2label[j]:v for j,v in enumerate(probs)}
        rows.append(dict(id=qid,expected=expected,structured_status=status,covered=sorted(covered),
                         primary_cells=sorted({events[i][1] for i in ids if events[i][3] in facets})
                         if status=='SUPPORTED' else [],
                         direct_available=grade.count('3'),direct_retrieved=sum(c['grade']==3 for c in candidates),
                         candidates=candidates))
    a.output.write_text(json.dumps(dict(discovery='dense+lexical RRF' if dense is not None else 'lexical only',
        nli='AUTHORED_CLAIMS' if nli is not None else 'NOT_RUN',rows=rows),indent=2)+'\n')


if __name__=='__main__':
    main()
