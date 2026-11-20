"""Frozen MoM native-cache routed-memory development study."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT.parent/'mom-int-001'),str(ROOT.parent/'ma-001/scripts')]
from preflight import MOM, REVISION, adapt_time_major_kernel
from run_experiment import Ranker, embeddings

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
ARMS=('no_memory','oracle','routed','fault')


def main():
    p=argparse.ArgumentParser()
    for key in ('mom_source','mom_snapshot','tokenizer_snapshot','e5_snapshot','output'):
        p.add_argument(key,type=Path)
    a=p.parse_args()
    a.output.mkdir(parents=True)
    def write(name,obj):
        (a.output/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    try:
        import torch
        from transformers import AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required')
        if subprocess.check_output(['git','-C',str(a.mom_source),'rev-parse','HEAD'],text=True).strip()!=MOM:
            raise ValueError('Incorrect MoM source')
        if a.mom_snapshot.name!=REVISION:
            raise ValueError('Expected pinned Hugging Face MoM snapshot path')
        sys.path.insert(0,str(a.mom_source.resolve()))
        from mom.models.mom_gated_deltanet import MomGatedDeltaNetForCausalLM
        from mom.layers import mom_gated_deltanet as layer
        for name in ('chunk_gated_delta_rule','fused_recurrent_gated_delta_rule'):
            setattr(layer,name,adapt_time_major_kernel(getattr(layer,name)))
        tok=AutoTokenizer.from_pretrained(a.tokenizer_snapshot,local_files_only=True)
        model,info=MomGatedDeltaNetForCausalLM.from_pretrained(a.mom_snapshot,local_files_only=True,
                         torch_dtype=torch.float16,output_loading_info=True)
        if any(info.values()) or len(tok)!=model.config.vocab_size:
            raise ValueError(f'Unexpected weight loading or vocabulary: {info}')
        model=model.cuda().eval().requires_grad_(False)
        encode=embeddings(False,a.output,a.e5_snapshot)
    except Exception as exc:
        write('decision.json',dict(status='BLOCKED',reason=repr(exc)))
        return 2

    records=[]
    labels=[]
    for i,(cell,label,value,wrong) in enumerate(FACTS):
        for neg,text in [(False,f'The {label} for the current study is {value}.'),
                         (True,f'In a different trial, the {label} was {wrong}.')]:
            records.append(dict(id=len(records),cell_id=(FACTS[(i+3)%9][0] if neg else cell),
                raw=text,summary=label,entities=[label],topics=[],aliases=[]))
        labels.append(label)
    cases=[(f'What is the current {x}?',[i]) for i,x in enumerate(labels)]
    cases.extend((f'Give the current {labels[i]} and {labels[j]}, in that order.',[i,j])
                 for i,j in [(0,3),(2,6),(4,7)])
    cases.extend((q,[]) for q in ('Who approved the current compiler marker?',
          'What is the password for the dataset snapshot?', 'What is the boiling point of mercury?'))
    ranker=Ranker(records,raw=True)
    frozen_path=ROOT.parent/'ma-001/artifacts/enriched-cpu/frozen.json'
    threshold=json.loads(frozen_path.read_text())['thresholds']['E3-Raw']
    dense=(encode(['query: '+q for q,_ in cases]) @ encode(['passage: '+r['raw'] for r in records]).T).tolist()

    def ids(text):
        return tok.encode(text,add_special_tokens=False)
    @torch.inference_mode()
    def forward(tokens,cache=None):
        result=model.model(input_ids=torch.tensor([tokens],device='cuda'),past_key_values=cache,
                           use_cache=True,return_dict=True)
        return result.last_hidden_state,result.past_key_values

    def score(query,context,answer):
        prefix=[tok.bos_token_id]+ids('Memory:\n'+'\n'.join(records[i]['raw'] for i in context))
        _,cache=forward(prefix)
        prompt=ids('\nQuestion: '+query+'\nAnswer:')
        target=ids(' '+answer)
        hidden,_=forward(prompt+target[:-1],cache)
        with torch.inference_mode():
            logits=model.lm_head(hidden[:,len(prompt)-1:]).float().log_softmax(-1)
            return float(logits[0,range(len(target)),target].mean()),len(target)

    rows=[]
    write('manifest.json',dict(mom_source=MOM,model_revision=REVISION,
          tokenizer_snapshot=str(a.tokenizer_snapshot),threshold=threshold,
          frozen_config_sha256=hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
          records=records,cases=cases))
    for (query,targets),ds in zip(cases,dense):
        # One synthetic Cell per address record lets the unchanged Ranker expose
        # its record scores before the normal max-per-Cell aggregation.
        indexed=Ranker([dict(r,cell_id=f'{i:02d}') for i,r in enumerate(records)],raw=True)
        ranked=indexed.scores(query,'E3-Raw',ds)[:16]
        route=[int(i) for i,_ in ranked]
        row=dict(query=query,targets=targets,candidates=[dict(**records[int(i)],rank=j,rrf=s)
                 for j,(i,s) in enumerate(ranked,1)],
                 threshold_cells=ranker.route(ranker.scores(query,'E3-Raw',ds),threshold),
                 arms={})
        if targets:
            good=[2*i for i in targets]
            wrong=2*targets[0]+1
            answer=', '.join(FACTS[i][2] for i in targets)
            distractor=', '.join([FACTS[targets[0]][3]]+[FACTS[i][2] for i in targets[1:]])
            row.update(coverage=set(good).issubset(route),fault_record=wrong)
            for arm,context in zip(ARMS,([],good,route,good+[wrong])):
                try:
                    yes,n=score(query,context,answer)
                    no,m=score(query,context,distractor)
                    row['arms'][arm]=dict(margin=yes-no,correct_logp=yes,distractor_logp=no,
                                           answer_tokens=[n,m])
                except (RuntimeError,ValueError) as exc:
                    row['arms'][arm]=dict(error=repr(exc))
        rows.append(row)
        write('queries.json',rows)
    valid=rows[:12]
    if any('margin' not in r['arms'][arm] for r in valid for arm in ARMS):
        write('decision.json',dict(status='BLOCKED',reason='Incomplete scored conditions'))
        return 2
    positive=lambda r,arm:r['arms'][arm]['margin']>0
    oracle=sum(positive(r,'oracle') for r in valid)
    routed=sum(positive(r,'routed') for r in valid)
    flips=sum(positive(r,'oracle') and not positive(r,'fault') for r in valid)
    multi=sum(r['coverage'] for r in valid[9:])
    write('decision.json',dict(status='NO_GO' if oracle<10 else 'GO' if routed>=9 and flips<=2 and multi>=2
        else 'PARTIAL_PASS',oracle_correct=oracle,routed_correct=routed,fault_flips=flips,
        oracle_denominator=oracle,route_coverage=sum(r['coverage'] for r in valid),multi_coverage=multi,
        weak_top16_exposed=sum(bool(r['candidates']) for r in rows[12:14]),
        weak_threshold_routes=sum(bool(r['threshold_cells']) for r in rows[12:14]),
        memory_gain={arm:sum(r['arms'][arm]['margin']-r['arms']['no_memory']['margin']
                            for r in valid)/12 for arm in ARMS[1:]}))
    return 0


if __name__=='__main__':
    sys.exit(main())
