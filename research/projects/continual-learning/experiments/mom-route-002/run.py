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
from run_experiment import Registry, Ranker, embeddings

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
E5_REV='614241f622f53c4eeff9890bdc4f31cfecc418b3'
ARMS=('no_memory','oracle','routed','legacy_top16')


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
        if a.tokenizer_snapshot.name!=TOKENIZER_REV or a.e5_snapshot.name!=E5_REV:
            raise ValueError('Unexpected tokenizer or E5 snapshot revision')
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

    registry=Registry(a.output/'registry.sqlite')
    for cell in sorted({f[0] for f in FACTS}):
        registry.create(cell,'projects/'+cell,cell,'fixture')
    records=[]
    labels=[]
    for i,(cell,label,value,wrong) in enumerate(FACTS):
        for neg,text in [(False,f'The {label} for the current study is {value}.'),
                         (True,f'In a different trial, the {label} was {wrong}.')]:
            records.append(dict(id=len(records),cell_id=(FACTS[(i+3)%9][0] if neg else cell),
                raw=text,summary=label,entities=[label],topics=[],aliases=[]))
        labels.append(label)
    for r in records:
        registry.write(f"{r['id']:02}",r['raw'],
            {k:r[k] for k in ('summary','entities','topics','aliases')},
            explicit=['projects/'+r['cell_id']])
    records=[dict(r,id=int(r['canonical_event_id'])) for r in registry.records()]
    registry.db.close()
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
        native=[torch.topk(t,model.config.topk,dim=-1).indices.cpu().tolist()
                for t in result.router_logits if t is not None]
        return result.last_hidden_state,result.past_key_values,native

    def score(query,context,answer):
        prefix=[tok.bos_token_id]+ids('Memory:\n'+'\n'.join(f"[Cell: {records[i]['cell_id']}]\n[Record: {i}]\n{records[i]['raw']}" for i in context))
        _,cache,native=forward(prefix)
        prompt=ids('\nQuestion: '+query+'\nAnswer:')
        target=ids(' '+answer)
        hidden,_,query_native=forward(prompt+target[:-1],cache)
        with torch.inference_mode():
            logits=model.lm_head(hidden[:,len(prompt)-1:]).float().log_softmax(-1)
            return float(logits[0,range(len(target)),target].mean()),len(target),dict(prefill=native,query=query_native)

    rows=[]
    write('manifest.json',dict(mom_source=MOM,model_revision=REVISION,
          tokenizer_snapshot=str(a.tokenizer_snapshot),threshold=threshold,
          frozen_config_sha256=hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
          tokenizer_revision=TOKENIZER_REV,e5_revision=E5_REV,
          records=records,cases=cases))
    answer_checks=[]
    for _,targets in cases[:12]:
        yes=', '.join(FACTS[i][2] for i in targets)
        no=', '.join([FACTS[targets[0]][3]]+[FACTS[i][2] for i in targets[1:]])
        y,n=ids(' '+yes),ids(' '+no)
        if not y or not n or y==n:
            raise ValueError('Invalid answer tokenization')
        answer_checks.append(dict(correct=yes,distractor=no,correct_ids=y,distractor_ids=n,
                                  equal_lengths=len(y)==len(n)))
    write('tokenizer-check.json',answer_checks)
    for qi,((query,targets),ds) in enumerate(zip(cases,dense)):
        print(f'Query {qi+1}/15',flush=True)
        # One synthetic Cell per address record lets the unchanged Ranker expose
        # its record scores before the normal max-per-Cell aggregation.
        indexed=Ranker([dict(r,cell_id=f'{i:02d}') for i,r in enumerate(records)],raw=True)
        ranked=indexed.scores(query,'E3-Raw',ds)[:16]
        route=[int(i) for i,_ in ranked]
        row=dict(query=query,targets=targets,candidates=[dict(**records[int(i)],rank=j,rrf=s)
                 for j,(i,s) in enumerate(ranked,1)],
                 threshold_cells=ranker.route(ranker.scores(query,'E3-Raw',ds),threshold),
                 arms={})
        selected=set(row['threshold_cells'])
        actual=[r['id'] for r in records if r['cell_id'] in selected]
        required={FACTS[i][0] for i in targets}
        row.update(selected_record_ids=actual,excluded_cell_ids=sorted({r['cell_id'] for r in records}-selected),
            cell_recall=len(required & selected)/len(required) if required else None,
            cell_precision=len(required & selected)/len(selected) if selected else None,
            evidence_missing=not targets or not set(2*i for i in targets).issubset(actual),
            leakage=sum(records[i]['cell_id'] not in selected for i in actual))
        if targets:
            good=[2*i for i in targets]
            wrong=2*targets[0]+1
            answer=', '.join(FACTS[i][2] for i in targets)
            distractor=', '.join([FACTS[targets[0]][3]]+[FACTS[i][2] for i in targets[1:]])
            row.update(coverage=set(good).issubset(route),fault_record=wrong)
            conditions=dict(zip(ARMS,([],good,actual,route)))
            if qi in (0,1,5,6,9):
                conditions.update(wrong_only=[wrong],correct_then_wrong=good+[wrong],wrong_then_correct=[wrong]+good)
            for arm,context in conditions.items():
                try:
                    yes,n,native=score(query,context,answer)
                    no,m,distractor_native=score(query,context,distractor)
                    row['arms'][arm]=dict(margin=yes-no,correct_logp=yes,distractor_logp=no,
                                           record_ids=context,answer_tokens=[n,m],native_router=native,distractor_native_router=distractor_native)
                except (RuntimeError,ValueError) as exc:
                    row['arms'][arm]=dict(error=repr(exc))
        rows.append(row)
        write('queries.json',rows)
    valid=rows[:12]
    if any('margin' not in r['arms'][arm] for r in valid for arm in r['arms']):
        write('decision.json',dict(status='BLOCKED',reason='Incomplete scored conditions'))
        return 2
    positive=lambda r,arm:r['arms'][arm]['margin']>0
    oracle=sum(positive(r,'oracle') for r in valid)
    routed=sum(positive(r,'routed') for r in valid)
    probes=[r for r in valid if 'correct_then_wrong' in r['arms']]
    flips={arm:sum(positive(r,'oracle') and r['arms'][arm]['margin']<0 for r in probes)
           for arm in ('correct_then_wrong','wrong_then_correct')}
    breakdown={}
    for name,group in [('single',valid[:9]),('multi',valid[9:]),('weak',rows[12:14]),('no_match',rows[14:])]:
        breakdown[name]=dict(count=len(group),selected_counts=[len(r['threshold_cells']) for r in group],
            evidence_missing=sum(r['evidence_missing'] for r in group),leakage=sum(r['leakage'] for r in group),
            accuracy_counts={arm:sum(positive(r,arm) for r in group) for arm in ARMS} if group[0]['targets'] else None)
    multi=sum(r['cell_recall']==1 for r in valid[9:])
    write('decision.json',dict(status='NO_GO' if oracle<10 else 'GO' if routed>=9 and max(flips.values())<=2 and multi>=2 and not any(r['leakage'] for r in rows)
        else 'PARTIAL_PASS',oracle_correct=oracle,routed_correct=routed,fault_flips=flips,breakdown=breakdown,legacy_correct=sum(positive(r,'legacy_top16') for r in valid),
        no_match_false_routes=sum(bool(r['threshold_cells']) for r in rows[14:]),
        oracle_denominator=len(valid),fault_oracle_denominator=sum(positive(r,"oracle") for r in probes),route_coverage=sum(r['cell_recall']==1 for r in valid),multi_coverage=multi,
        weak_top16_exposed=sum(bool(r['candidates']) for r in rows[12:14]),
        weak_threshold_routes=sum(bool(r['threshold_cells']) for r in rows[12:14]),
        memory_gain={arm:sum(r['arms'][arm]['margin']-r['arms']['no_memory']['margin']
                            for r in valid)/12 for arm in ARMS[1:]}))
    return 0


if __name__=='__main__':
    sys.exit(main())
