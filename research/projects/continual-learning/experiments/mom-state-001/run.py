"""Oracle-only shared native cache; independent writer and restore processes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'mom-int-001'))
from preflight import MOM, REVISION, adapt_time_major_kernel, restore_cache, compare
FACTS=[
('minijam','test gateway alias','veka-27','nalo-27'),
('minijam','peer relay tag','voro-63','sena-63'),
('minijam','RPC probe marker','tiva-51','mora-51'),
('jamscript','compiler marker','sena-41','veka-41'),
('jamscript','source map label','nalo-82','voro-82'),
('jamscript','service runtime tag','mora-34','tiva-34'),
('minicells','dataset snapshot ID','voro-62','sena-62'),
('minicells','training run alias','tiva-53','mora-53'),
('minicells','model checkpoint label','nalo-78','veka-78')]
TOKENIZER_REV='46b15820a4df269e99aed9d709e017677c15d24b'
from state import clone_tree, export, fingerprint, query_copy

CASES=[(f'What is the current {f[1]}?',[i]) for i,f in enumerate(FACTS)]
CASES += [(f'Give the current {FACTS[i][1]} and {FACTS[j][1]}, in that order.',[i,j])
          for i,j in ((0,3),(2,6),(4,7))]
NEW=('What is the current export verification marker?', 'zena-85','mora-85')


def main():
    p=argparse.ArgumentParser()
    for name in ('source','snapshot','tokenizer','output'):
        p.add_argument(name,type=Path)
    p.add_argument('--phase',choices=('all','write','restore'),default='all')
    a=p.parse_args()
    out=a.output.resolve()
    def save(name,data):
        (out/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    if a.phase=='all':
        out.mkdir(parents=True)
        try:
            for phase in ('write','restore'):
                subprocess.run([sys.executable,__file__,str(a.source),str(a.snapshot),
                                str(a.tokenizer),str(out),'--phase',phase],check=True)
        except subprocess.CalledProcessError as exc:
            save('decision.json',dict(status='BLOCKED',phase=phase,returncode=exc.returncode))
            return 2
        return 0

    import torch
    from transformers import AutoTokenizer
    import transformers
    if subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip()!=MOM:
        raise ValueError('Wrong MoM source')
    if a.snapshot.name!=REVISION or a.tokenizer.name!=TOKENIZER_REV:
        raise ValueError('Wrong model or tokenizer snapshot')
    sys.path.insert(0,str(a.source.resolve()))
    from mom.models.mom_gated_deltanet import MomGatedDeltaNetForCausalLM
    from mom.layers import mom_gated_deltanet as layer
    from fla.models.utils import Cache
    for name in ('chunk_gated_delta_rule','fused_recurrent_gated_delta_rule'):
        setattr(layer,name,adapt_time_major_kernel(getattr(layer,name)))
    torch.manual_seed(42)
    started=time.monotonic()
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    model,info=MomGatedDeltaNetForCausalLM.from_pretrained(a.snapshot,local_files_only=True,
                               torch_dtype=torch.float16,output_loading_info=True)
    if any(info.values()) or len(tok)!=model.config.vocab_size:
        raise ValueError(f'Model load or tokenizer mismatch: {info}')
    model=model.cuda().eval().requires_grad_(False)
    identity=dict(model=REVISION,mom=MOM,tokenizer=TOKENIZER_REV,schema=1)
    save(f'environment-{a.phase}.json',dict(**identity,torch=str(torch.__version__),
        transformers=transformers.__version__,gpu=torch.cuda.get_device_name()))
    hashes=[]
    def trace(name,row):
        with (out/f'{a.phase}-{name}.jsonl').open('a') as f:
            f.write(json.dumps(row,allow_nan=False)+'\n')
    def ids(s):
        return tok.encode(s,add_special_tokens=False)
    @torch.inference_mode()
    def forward(tokens,cache):
        return model.model(input_ids=torch.tensor([tokens],device='cuda'),
                           past_key_values=cache,use_cache=True,return_dict=True)
    def append(cache,text,record_ids):
        begin=time.monotonic()
        tokens=ids(text)
        if cache is None:
            tokens=[tok.bos_token_id]+tokens
        r=forward(tokens,cache)
        state=export(r.past_key_values,torch)
        digest=fingerprint(state,torch)
        trace('writes',dict(record_ids=record_ids,sha256=digest,position=state['seen_tokens'],
            seconds=time.monotonic()-begin,
            experts=[torch.topk(t,model.config.topk,dim=-1).indices.flatten().bincount(
                       minlength=model.config.num_memories).tolist() for t in r.router_logits if t is not None]))
        hashes.append(digest)
        return r.past_key_values
    @torch.inference_mode()
    def ask(cache,query,yes,no):
        before=export(cache,torch)
        prompt=ids('\nQuestion: '+query+'\nAnswer:')
        def score(answer):
            target=ids(' '+answer)
            r=forward(prompt+target[:-1],query_copy(cache,torch,restore_cache))
            logits=model.lm_head(r.last_hidden_state[:,len(prompt)-1:]).float().log_softmax(-1)
            return float(logits[0,range(len(target)),target].mean())
        a,b=score(yes),score(no)
        after=export(cache,torch)
        h1,h2=fingerprint(before,torch),fingerprint(after,torch)
        return dict(query=query,correct_logp=a,distractor_logp=b,margin=a-b,
                    before=h1,after=h2,read_only=h1==h2 and not compare(before,after,torch))
    def assess(cache,label,cases):
        rows=[]
        for q,targets in cases:
            try:
                if targets:
                    yes=', '.join(FACTS[i][2] for i in targets)
                    no=', '.join([FACTS[targets[0]][3]]+[FACTS[i][2] for i in targets[1:]])
                else:
                    _,yes,no=NEW
                row=ask(cache,q,yes,no)
            except (RuntimeError,ValueError) as exc:
                row=dict(query=q,error=repr(exc))
            trace('queries',dict(stage=label,**row))
            rows.append(row)
        print(a.phase,label,len(rows),flush=True)
        return rows
    bundle=out/'cache.pt'
    def digest():
        return hashlib.sha256(bundle.read_bytes()).hexdigest()
    def append_d(cache):
        return append(cache,'\n[Cell: minicells]\nThe current export verification marker is zena-85.', ['D'])
    if a.phase=='write':
        cache=None
        for i,cell in enumerate(('minijam','jamscript','minicells')):
            text='\n'.join(f'[Cell: {c}]\n[Record: {j}]\nThe {label} for the current study is {v}.'
                           for j,(c,label,v,_) in enumerate(FACTS) if c==cell)
            cache=append(cache,'\nMemory:\n'+text,list(range(i*3,(i+1)*3)))
            assess(cache,f'after-{cell}',CASES[:(i+1)*3] if i<2 else [])
        before=assess(cache,'ABC',CASES)
        torch.save(export(cache,torch),bundle)
        save('checkpoint.json',dict(identity=identity,sha256=digest(),record_ids=list(range(9)),
                                   cell_versions=dict(minijam=3,jamscript=3,minicells=3)))
        cache=append_d(cache)
        after=assess(cache,'ABCD',CASES+[(NEW[0],[])])
        torch.save(export(cache,torch),out/'reference-final.pt')
        save('reference.json',dict(ABC=before,ABCD=after))
        return 0

    manifest=json.loads((out/'checkpoint.json').read_text())
    if manifest['identity']!=identity or manifest['sha256']!=digest():
        raise ValueError('Checkpoint integrity or identity mismatch')
    saved=torch.load(bundle,weights_only=True)
    cache=restore_cache(Cache,clone_tree(saved['layers'],torch,'cuda'),saved['seen_tokens'])
    initial=compare(saved,export(cache,torch),torch)
    before=assess(cache,'ABC',CASES)
    cache=append_d(cache)
    after=assess(cache,'ABCD',CASES+[(NEW[0],[])])
    final=compare(torch.load(out/'reference-final.pt',weights_only=True),export(cache,torch),torch)
    ref=json.loads((out/'reference.json').read_text())
    scores=[]
    for name,rows in [('ABC',before),('ABCD',after)]:
        for x,y in zip(ref[name],rows,strict=True):
            scores.append(dict(stage=name,query=y['query'],matches=all(k in x and k in y and x[k]==y[k]
                    for k in ('correct_logp','distractor_logp')),
                    abs_deltas={k:abs(x[k]-y[k]) for k in ('correct_logp','distractor_logp') if k in x and k in y}))
    save('restore_comparison.json',dict(initial=initial,continued=final,scores=scores,atol=0,rtol=0))
    allrows=before+after+ref['ABC']+ref['ABCD']
    engineering=not initial and not final and all(r['matches'] for r in scores) and all(r.get('read_only') for r in allrows)
    save('decision.json',dict(status='BASELINE_COMPLETE' if engineering else 'BASELINE_MISMATCH',
        scope='ORACLE_SEQUENTIAL_BASELINE_ONLY',seconds=time.monotonic()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),restored_abc_positive=sum(r.get('margin',0)>0 for r in before),
        restored_abcd_positive=sum(r.get('margin',0)>0 for r in after),
        development_only=True))
    return 0


if __name__=='__main__':
    sys.exit(main())
