"""CPU experiment. Pass --download explicitly to acquire the pinned encoder."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from addressing import Registry, Ranker

MODEL='intfloat/multilingual-e5-small'
REV='614241f622f53c4eeff9890bdc4f31cfecc418b3'
ARMS=('E0','E1','E2','E3-Raw','E3-Descriptor')


def load(name):
    return [json.loads(line) for line in (ROOT/'fixtures'/f'{name}.jsonl').read_text().splitlines()]


def dump(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def embeddings(download, output, snapshot=None):
    import torch
    import transformers
    torch.manual_seed(42)
    torch.set_num_threads(4)
    kwargs=dict(revision=REV,local_files_only=not download)
    source = snapshot or MODEL
    tok=transformers.AutoTokenizer.from_pretrained(source,**kwargs)
    model=transformers.AutoModel.from_pretrained(source,**kwargs).eval()
    dump(output/'encoder.json',dict(model=MODEL,revision=REV,torch=str(torch.__version__),
                                  transformers=transformers.__version__,
                                  local_assets={f.name:hashlib.sha256(f.read_bytes()).hexdigest()
                                                for f in sorted(snapshot.iterdir()) if f.is_file()} if snapshot else None))

    @torch.no_grad()
    def encode(text):
        identity=hashlib.sha256(json.dumps([MODEL,REV,text]).encode()).hexdigest()
        path=output/f'embeddings-{identity}.pt'
        if path.exists():
            return torch.load(path,weights_only=True)
        chunks=[]
        for i in range(0,len(text),16):
            batch=tok(text[i:i+16],padding=True,truncation=True,max_length=512,return_tensors='pt')
            h=model(**batch).last_hidden_state
            mask=batch.attention_mask.unsqueeze(-1)
            chunks.append(torch.nn.functional.normalize((h*mask).sum(1)/mask.sum(1),dim=-1))
        matrix=torch.cat(chunks)
        torch.save(matrix,path)
        return matrix
    return encode


def evaluate(rows, queries, arm, dense, threshold):
    result=[]
    for i, q in enumerate(queries):
        ranking=rows.scores(q['query'],arm,None if dense is None else dense[i])
        returned=rows.route(ranking,threshold)
        expected=set(q['relevant_cell_ids'])
        result.append(dict(**q, returned_cell_ids=returned,
                           recall=len(expected.intersection(returned))/len(expected) if expected else None,
                           all_relevant=expected.issubset(returned) if expected else None))
    return result


def metrics(results):
    mean=lambda xs:sum(xs)/len(xs) if xs else None
    pos=[r for r in results if r['recall'] is not None]
    multi=[r for r in pos if r['query_type']=='multi']
    return dict(recall_at_3=mean([r['recall'] for r in pos]),
                multi_recall_at_3=mean([r['recall'] for r in multi]),
                multi_all_relevant_at_3=mean([r['all_relevant'] for r in multi]),
                no_match_fpr=mean([bool(r['returned_cell_ids']) for r in results if r['query_type']=='no_match']),
                insufficient_fpr=mean([bool(r['returned_cell_ids']) for r in results if r['query_type']=='insufficient']),
                by_type={kind:mean([r['recall'] for r in pos if r['query_type']==kind])
                         for kind in ('direct','paraphrase','multi','hard')})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['all','write','retrieval','cross-session','persistence'],default='all')
    p.add_argument('--output',type=Path,default=ROOT/'artifacts'/'local')
    p.add_argument('--snapshot',type=Path,help='Local copy of the pinned encoder snapshot')
    p.add_argument('--download',action='store_true')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if a.mode in ('all','persistence'):
        subprocess.run([sys.executable,'-m','unittest','discover','-s','tests'],cwd=ROOT,check=True)
    db=a.output/'registry.sqlite'
    if db.exists():
        raise FileExistsError('Select a fresh output directory to keep run evidence')
    reg=Registry(db)
    for c in load('cells'):
        reg.create(c['id'],c['category'],c['id'],c['session'])
    expected={c['category']:c['id'] for c in load('cells')}
    for e in load('events'):
        assert reg.write(e['id'],e['raw'],e['descriptor'],explicit=e['targets'])==[expected[t] for t in e['targets']]
    reg.write('projection','GPU-PVM and portable learned cells',
              dict(summary='joint execution review',entities=['joint execution review'],topics=[],aliases=[]),
              explicit=['minijam','minicells'])
    records=reg.records()
    reg.db.close()
    dump(a.output/'address_records.json',records)
    hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((ROOT/'fixtures').glob('*.jsonl'))}
    dump(a.output/'manifest.json',dict(seed=42,datasets=hashes,
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        write_assignment_accuracy=1.0))
    if a.mode in ('all','cross-session','persistence'):
        # The child starts with only the database path; it has no writer or session state.
        syscode = '''
import json,sys
from addressing import Registry
r = Registry(sys.argv[1])
print(json.dumps(r.records(), ensure_ascii=False))
r.db.close()
'''
        recovered=json.loads(subprocess.check_output([sys.executable,'-c',syscode,str(db.resolve())],
                                                     cwd=ROOT,text=True))
        if recovered != records:
            raise RuntimeError('Independent process index readback changed records')
        dump(a.output/'persistence.json',dict(status='PASS',fresh_process=True,records=len(recovered)))
    if a.mode in ('write','persistence'):
        return
    try:
        encode=embeddings(a.download,a.output,a.snapshot)
    except (ImportError,OSError) as error:
        dump(a.output/'decision.json',dict(status='BLOCKED',reason=str(error),scientific='NOT_EVALUATED'))
        print('BLOCKED:',error)
        return
    dev=load('queries.dev')
    dev_vectors=encode(['query: '+q['query'] for q in dev])
    arms={}
    for arm in ARMS:
        ranker=Ranker(records,raw=arm=='E3-Raw')
        passages=encode(['passage: '+s for s in ranker.text]) if arm not in ('E0','E2') else None
        dense=(dev_vectors@passages.T).tolist() if passages is not None else None
        values={s for i,q in enumerate(dev) for _,s in ranker.scores(q['query'],arm,None if dense is None else dense[i])}
        def score(threshold):
            rows=evaluate(ranker,dev,arm,dense,threshold)
            return (sum(set(r['returned_cell_ids'])==set(r['relevant_cell_ids']) for r in rows),threshold)
        threshold=max(map(score,values|{0.}))[1]
        arms[arm]=(ranker,passages,threshold)
    dump(a.output/'frozen.json',dict(thresholds={k:v[2] for k,v in arms.items()},datasets=hashes,revision=REV))
    if a.mode in ('all','cross-session'):
        query=next(q for q in load('queries.dev')+load('queries.holdout') if q['query_id']=='q033')
        ranker,passages,threshold=arms['E3-Descriptor']
        returned=evaluate(ranker,[query],'E3-Descriptor',
                          (encode(['query: '+query['query']])@passages.T).tolist(),threshold)[0]
        dump(a.output/'cross-session.json',dict(session_id='session_c',
             expected_cell_ids=query['relevant_cell_ids'],returned_cell_ids=returned['returned_cell_ids'],
             status='PASS' if returned['all_relevant'] else 'FAIL'))
        if a.mode=='cross-session':
            return
    holdout=load('queries.holdout')
    vectors=encode(['query: '+q['query'] for q in holdout])
    summary={}
    for arm,(ranker,passages,threshold) in arms.items():
        dense=(vectors@passages.T).tolist() if passages is not None else None
        rows=evaluate(ranker,holdout,arm,dense,threshold)
        dump(a.output/f'{arm}.json',rows)
        summary[arm]=metrics(rows)
    dump(a.output/'metrics.json',summary)
    dump(a.output/'decision.json',dict(status='REVIEW_REQUIRED',scientific='NOT_ADJUDICATED'))


if __name__=='__main__':
    main()
