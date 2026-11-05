"""Frozen C1 forward-only interventions. No optimizer, backward, or training fallback."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess
import time
import platform
import numpy as np
import torch
from torch.nn import functional as F
from frozen_model import (AssetUnavailable, BaselineIntegrityError, check_deadline,
                          recover_frozen_assets, load_frozen_model)
from runtime import Runtime, weight_hash
from native_clm_runtime import sha256_file


class CompletedDiagnostic(Exception):
    """Exit recovery directly to the common publication path."""


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def functional_preflight():
    """Tiny forward-only CPU gate, independent of private model assets."""
    from frozen_model import DiagnosticCore
    from model import Config
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(713)
        model=DiagnosticCore(Config(vocab=16,cells=4,dim=8)).eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
        state=torch.arange(32,dtype=torch.float32).reshape(1,4,8)/32
        before=state.clone(); branch=state.clone();branch[:,1]=0
        model.step(branch,[2]);model.read(state)
        if not torch.equal(state,before): raise BaselineIntegrityError('CPU state isolation/pure-read gate failed')
        normalized=model.norm(state)
        expected=state+.1*model.update(torch.cat((normalized,model.center(normalized)),-1))
        if not torch.equal(model.evolve(state,1,'CROSS_CELL_REMOVED'),expected):
            raise BaselineIntegrityError('CPU connection intervention gate failed')
    return 'PASS_FORWARD_ONLY_NO_TRAINING'


def sync(model):
    if next(model.parameters()).is_cuda: torch.cuda.synchronize()


def checked_step(model, state, token, steps=4, communication=True):
    state, logits = model.step(state, [token], steps, communication)
    if not torch.isfinite(state).all() or not torch.isfinite(logits).all():
        raise FloatingPointError('non-finite diagnostic state/logits')
    return state, logits


def loss(logits, target):
    value = F.cross_entropy(logits, torch.tensor([target],device=logits.device))
    if not torch.isfinite(value): raise FloatingPointError('non-finite diagnostic NLL')
    return float(value)


def aggregate(rows):
    tokens = sum(r['tokens'] for r in rows)
    nll = sum(r['nll_sum'] for r in rows)/tokens
    return {'nll':nll,'ppl':math.exp(min(nll,700)),'documents':len(rows),'tokens':tokens,'stories':rows}


def comparison(intervention, baseline, plan):
    a, b = intervention['stories'], baseline['stories']
    if [(r['story_id'],r['tokens']) for r in a] != [(r['story_id'],r['tokens']) for r in b]:
        raise BaselineIntegrityError('unpaired diagnostic documents')
    differences = np.array([x['nll_sum']-y['nll_sum'] for x,y in zip(a,b)])
    counts = np.array([x['tokens'] for x in a])
    samples = np.random.default_rng(plan['seed']).integers(len(a),size=(plan['bootstrap_draws'],len(a)))
    draws = differences[samples].sum(1)/counts[samples].sum(1)
    return {'delta_nll':float(differences.sum()/counts.sum()),
            'paired_document_mean_delta_nll':float(np.mean(differences/counts)),
            'document_bootstrap_95':np.quantile(draws,[.025,.975]).tolist(),
            'documents':len(a),'tokens':int(counts.sum()),
            'positive_delta_document_fraction':float(np.mean(differences>0)),
            'interpretation':'exploratory single-seed development; positive means intervention worsens prediction'}


def run_documents(model, documents, deadline, mode='NORMAL', steps=4, probes=None, fractions=(.25,.5,.75)):
    rows=[]; norms=[]; deltas=[]; sync(model); start=time.monotonic()
    for doc in documents:
        state=model.init_state(1); total=0.; tail_total=0.; targets=doc['tokens'][1:]
        midpoint=len(targets)//2
        for position,(token,target) in enumerate(zip(doc['tokens'],targets)):
            check_deadline(deadline)
            if probes is not None and position in {int(len(targets)*f) for f in fractions}:
                probes.append({'story_id':doc['id'],'position':position,'state':state.clone(),
                               'token':token,'target':target})
            if mode=='RESET_EVERY_TOKEN' or (mode=='RESET_AT_PROBE' and position==midpoint):
                state=model.reset(state)
            if mode=='ENTRY_ONLY': state=torch.cat((state[:,:1],torch.zeros_like(state[:,1:])),1)
            before=state
            communication=False if mode=='NEIGHBORS_REPLACED_BY_SELF' else 'CROSS_CELL_REMOVED' if mode=='CROSS_CELL_REMOVED' else True
            state,logits=checked_step(model,state,token,steps,communication)
            token_loss=loss(logits,target); total+=token_loss
            if position>=midpoint: tail_total+=token_loss
            norms.append(float(state.norm())); deltas.append(float((state-before).norm()))
        rows.append({'story_id':doc['id'],'nll_sum':total,'tokens':len(targets),'nll':total/len(targets),'probe_tail_sum':tail_total,'probe_tail_tokens':len(targets)-midpoint})
    sync(model); elapsed=time.monotonic()-start
    return {**aggregate(rows),'seconds':elapsed,'seconds_per_token':elapsed/sum(r['tokens'] for r in rows),
            'state_norm_mean':float(np.mean(norms)),'state_norm_max':max(norms),
            'event_state_delta_mean':float(np.mean(deltas)), 'numerical_stability':'PASS',
            'timing_scope':'synchronized forward plus diagnostic finite checks/metric collection; excludes asset/network time',
            'mode':mode,'K':steps}


def integrity_checks(model, document, deadline):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise BaselineIntegrityError('model not frozen')
    tokens=document['tokens'][:min(32,len(document['tokens'])-1)]
    runtime=Runtime(model); state=model.init_state(1); outputs=[]
    for token in tokens:
        check_deadline(deadline); state,logits=checked_step(model,state,token); outputs.append(logits.clone())
    before=state.clone(); model.read(state); model.read(state)
    if not torch.equal(before,state): raise BaselineIntegrityError('readout mutated state')
    repeat=model.init_state(1)
    for index,token in enumerate(tokens):
        repeat,logits=checked_step(model,repeat,token)
        if not torch.equal(logits,outputs[index]): raise BaselineIntegrityError('repeat output mismatch')
    split=len(tokens)//2; segmented=model.init_state(1)
    for token in tokens[:split]: segmented,_=checked_step(model,segmented,token)
    snapshot=runtime.snapshot(segmented)
    restored=runtime.restore(snapshot)
    for token in tokens[split:]:
        segmented,original_logits=checked_step(model,segmented,token)
        restored,restored_logits=checked_step(model,restored,token)
        if not torch.equal(original_logits,restored_logits): raise BaselineIntegrityError('snapshot prediction mismatch')
    if not torch.equal(segmented,state) or not torch.equal(restored,state):
        raise BaselineIntegrityError('stream/segment/snapshot state mismatch')
    return dict.fromkeys(['frozen','pure_read','repeat','stream_segment','snapshot_restore'],'PASS')


def target_score(model,state,logits,ids,deadline):
    total=0.
    for index,token in enumerate(ids):
        check_deadline(deadline); total-=loss(logits,token)
        if index+1<len(ids): state,logits=checked_step(model,state,token)
    return total


def history_pairs(model,tokenizer,plan,deadline):
    """Balanced two-direction margins; unequal lengths/boundaries are explicitly excluded."""
    rows=[]; excluded=[]; pair_count=0
    for colors in [('red','blue'),('green','yellow'),('black','white'),('pink','brown')]:
        for name in ('Alice','Bob','Lily','Tom'):
            if pair_count>=plan['history_pairs']: break
            prefixes=[f'{name} put a {color} ball in the box. Then everyone went outside.' for color in colors]
            suffix=f' Later, {name} opened the box. The ball was'
            prefix_ids=[tokenizer.encode(p).ids for p in prefixes]
            suffix_ids=tokenizer.encode(suffix).ids
            targets=[tokenizer.encode(' '+c).ids for c in colors]
            valid=(len(prefix_ids[0])==len(prefix_ids[1]) and len(targets[0])==len(targets[1])
                   and all(tokenizer.encode(p+suffix).ids==ids+suffix_ids for p,ids in zip(prefixes,prefix_ids))
                   and all(tokenizer.encode(suffix+' '+c).ids==suffix_ids+t for c,t in zip(colors,targets)))
            if not valid:
                excluded.append({'name':name,'colors':colors,'prefix_lengths':list(map(len,prefix_ids)),
                                 'target_lengths':list(map(len,targets)),'reason':'unequal event/token lengths or unstable encoding boundary'})
                continue
            states=[]
            for ids in prefix_ids:
                state=model.init_state(1)
                # Match natural document starts, without exposing the target suffix.
                for token in [tokenizer.token_to_id('<bos>')]+ids:
                    check_deadline(deadline); state,_=checked_step(model,state,token)
                states.append(state)
            for direction in (0,1):
                scores={}
                for mode,start in [('normal',states[direction]),('reset',torch.zeros_like(states[direction])),('swapped',states[1-direction])]:
                    state=start.clone()
                    for token in suffix_ids:
                        check_deadline(deadline); state,logits=checked_step(model,state,token)
                    values=[target_score(model,state.clone(),logits,t,deadline) for t in targets]
                    margin=values[direction]-values[1-direction]
                    scores[mode]={'margin':margin,'correct_target_nll':-values[direction]/len(targets[direction]),
                                  'correct_target_log_probability':values[direction],'alternative_log_probability':values[1-direction]}
                rows.append({'pair':pair_count,'name':name,'colors':colors,'direction':direction,
                             'prefix_tokens':len(prefix_ids[direction]),'suffix_tokens':len(suffix_ids),
                             'target_tokens':len(targets[direction]),'suffix_sha256':hashlib.sha256(json.dumps(suffix_ids).encode()).hexdigest(),'scores':scores,
                             'direction_correct':scores['normal']['margin']>0 and scores['swapped']['margin']<0,
                             'swap_margin_change':scores['normal']['margin']-scores['swapped']['margin']})
            pair_count+=1
    if not rows: return {'status':'INCONCLUSIVE','pairs':0,'excluded':excluded,'rows':[]}
    means={mode:{key:float(np.mean([r['scores'][mode][key] for r in rows])) for key in ('margin','correct_target_nll')} for mode in ('normal','reset','swapped')}
    paired_success=[all(r['direction_correct'] for r in rows if r['pair']==i) for i in range(pair_count)]
    return {'status':'COMPLETE' if pair_count>=8 else 'INCONCLUSIVE','pairs':pair_count,'rows':rows,'excluded':excluded,'excluded_count':len(excluded),
            'means':means,'bidirectional_pair_success_fraction':float(np.mean(paired_success)),
            'direction_correct_fraction':float(np.mean([r['direction_correct'] for r in rows])),
            'suffix_ids_identical_within_pair':True,'balanced_two_directions':True,
            'semantic_signal_rule':'at least 8 accepted pairs, >=75% bidirectional pairs correct, normal correct-target NLL < reset; descriptive only'}


def distribution_change(original,alternative):
    a=original.log_softmax(-1); b=alternative.log_softmax(-1)
    return {'kl_normal_to_intervention':max(0.,float((a.exp()*(a-b)).sum())),
            'total_variation':float((a.exp()-b.exp()).abs().sum()/2)}


def event_probes(model,probes,plan,deadline):
    communication=[]; cells=[]; evolution=[]
    for probe in probes:
        check_deadline(deadline)
        before=probe['state']; token,target=probe['token'],probe['target']
        normal,logits=checked_step(model,before.clone(),token)
        baseline=loss(logits,target)
        label={'story_id':probe['story_id'],'position':probe['position']}
        changed,other=checked_step(model,before.clone(),token,communication='CROSS_CELL_REMOVED')
        communication.append({**label,'delta_nll':loss(other,target)-baseline,
                              'cell_state_difference':(changed-normal).norm(dim=-1)[0].cpu().tolist(),
                              'state_difference':float((changed-normal).norm()),**distribution_change(logits,other)})
        for cell in plan['probe_cells']:
            check_deadline(deadline)
            zeroed=before.clone(); old_norm=float(zeroed[:,cell].norm()); zeroed[:,cell]=0
            result,other=checked_step(model,zeroed,token)
            cells.append({**label,'cell':cell,'topology_distance':min(cell,model.config.cells-cell),
                          'pre_intervention_cell_norm':old_norm,'delta_nll':loss(other,target)-baseline,
                          'state_difference':float((result-normal).norm()),**distribution_change(logits,other)})
        for extra in (0,*plan['extra_evolve']):
            check_deadline(deadline); sync(model); start=time.monotonic()
            # Normal already injected the event once. Extra work never calls inject/step.
            result=model.evolve(normal.clone(),extra); other=model.read(result)
            if not torch.isfinite(result).all() or not torch.isfinite(other).all():
                raise FloatingPointError('non-finite extra evolution')
            value=loss(other,target); sync(model)
            evolution.append({**label,'extra_steps':extra,'total_steps':4+extra,'nll':value,
                              'gain':baseline-value,'state_delta':float((result-normal).norm()),
                              'state_norm':float(result.norm()),'extra_seconds':time.monotonic()-start,
                              **distribution_change(logits,other)})
    return {'single_event':communication,'cell_sensitivity':cells}, {'single_event_extra_evolution':evolution}


def private_to_public(value):
    if isinstance(value,dict):
        return {k:private_to_public(v) for k,v in value.items() if k not in ('stories','rows','excluded','single_event','cell_sensitivity','single_event_extra_evolution')}
    if isinstance(value,list): return [private_to_public(v) for v in value]
    return value


def probe_summary(rows,key):
    if not rows: return {'status':'NOT_EVALUATED'}
    return {'events':len(rows),'mean':float(np.mean([r[key] for r in rows])),
            'positive_fraction':float(np.mean([r[key]>0 for r in rows]))}


def make_plot(report,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,3,figsize=(13,4))
    state=report.get('D1',{}); connection=report.get('D2',{}); evolution=report.get('D3',{})
    for axis,section,names,title in [(axes[0],state,['RESET_EVERY_TOKEN','RESET_AT_PROBE','ENTRY_ONLY'],'State Causality'),
                                           (axes[1],connection,['NEIGHBORS_REPLACED_BY_SELF','CROSS_CELL_REMOVED'],'Connection Causality')]:
        available=[n for n in names if n in section]
        axis.bar(range(len(available)),[section[n]['comparison']['delta_nll'] for n in available])
        axis.set_xticks(range(len(available))); axis.set_xticklabels(available,rotation=35,ha='right',fontsize=7)
        axis.set(title=title,ylabel='Intervention minus normal NLL'); axis.axhline(0,color='gray',linewidth=.5)
    means=state.get('history_pairs',{}).get('means',{})
    if means:
        inset=axes[0].inset_axes([.54,.62,.43,.32])
        labels=['normal','reset','swapped'];reference=means['normal']['correct_target_nll']
        inset.bar(range(3),[means[k]['correct_target_nll']-reference for k in labels])
        inset.set_xticks(range(3));inset.set_xticklabels(labels,fontsize=6,rotation=25)
        inset.set_title('Controlled history ΔNLL',fontsize=7)
    available=[str(k) for k in (2,4,8) if str(k) in evolution]
    axes[2].plot(available,[evolution[k]['nll'] for k in available],marker='o',label='NLL')
    cost=axes[2].twinx();cost.plot(available,[evolution[k]['seconds_per_token']*1000 for k in available],color='orange',marker='x')
    axes[2].set(title='Internal Evolution',xlabel='K',ylabel='NLL');cost.set_ylabel('Measured ms/token',color='orange')
    if not report.get('D0'): figure.suptitle(report['status'])
    figure.tight_layout();figure.savefig(path);plt.close(figure)


def publish(root,output,report):
    from scripts.research.publish_experiment_results import prepare_artifacts,push_results,EXPERIMENTS
    publication=root/'results/epclm-core-001-c1-diagnostic/publish'
    publication.mkdir(parents=True,exist_ok=True)
    write(publication/'decision.json',private_to_public(report))
    for section,name in [('D0','baseline'),('D1','state-causality'),('D2','connection-causality'),('D3','internal-evolution')]:
        write(publication/(name+'.json'),private_to_public(report.get(section,{'status':'NOT_EVALUATED'})))
    summary='# C1 frozen-model mechanism diagnostic\n\n'+report['status']+'\n\n'
    summary+='Forward only; optimizer updates 0. H1–H5 NOT_EVALUATED. Single-seed development interventions do not confirm architectural superiority or reasoning.\n\n'
    summary+='Findings: '+', '.join(report.get('findings',[]))+'\n'
    if report.get('error'): summary+='\nStop reason: '+report['error']+'\n'
    (publication/'summary.md').write_text(summary)
    make_plot(report,publication/'mechanism-diagnostic.png')
    reports=output/'reports';reports.mkdir(exist_ok=True)
    for name in EXPERIMENTS['epclm-core-001-c1-diagnostic'].files:
        shutil.copyfile(publication/name,reports/name)
    destination=prepare_artifacts(root,'epclm-core-001-c1-diagnostic',EXPERIMENTS['epclm-core-001-c1-diagnostic'],None)
    push_results(root,destination,'epclm-core-001-c1-diagnostic','kaggle/epclm-core-001-c1-diagnostic-'+report['identity']['diagnostic_source'][:12],'GITHUB_TOKEN','origin/main')


@torch.inference_mode()
@torch.autocast(device_type='cuda', enabled=False)
def launch_diagnostic(root,base,output,device='cuda'):
    from scripts.research.publish_experiment_results import load_github_token,validate_origin
    from storage import bundle
    if device!='cuda' or not torch.cuda.is_available(): raise RuntimeError('enable Kaggle GPU')
    validate_origin(root); load_github_token('GITHUB_TOKEN')
    output.mkdir(parents=True,exist_ok=True)
    plan=json.loads((base/'diagnostic-protocol.json').read_text())
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    identity={'experiment':'epclm-core-001-c1-diagnostic','mode':'FROZEN_MODEL_DIAGNOSTIC',
              'training_source':plan['training_identity']['source'],
              'training_weight_sha256':plan['checkpoint_identity']['weights_sha256'],
              'training_checkpoint_sha256':plan['checkpoint_identity']['sha256'],
              'diagnostic_source':source,'diagnostic_protocol_sha256':sha256_file(base/'diagnostic-protocol.json'),
              'data_manifest_sha256':plan['training_identity']['data'],'training_tokens':250000,
              'optimizer_updates':0,'formal_confirmation':False,'interventions':plan}
    report_path=output/'decision.json'
    report={'format':'minicells.epclm-core-001.c1-diagnostic.v1','status':'INCOMPLETE','identity':identity,
            'hypotheses':dict.fromkeys(['H1','H2','H3','H4','H5'],'NOT_EVALUATED'),
            'findings':[],'timings':{},'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(),'precision':'fp32','autocast_enabled':False,'grad_scaler_created':False,'inference_mode':True},
            'limitations':['single training seed; 16 development documents; out-of-distribution interventions',
                          'no independently trained no-connection control; no evidence of reasoning or universal architectural superiority']}
    if report_path.exists():
        previous=json.loads(report_path.read_text())
        if previous['identity']!=identity: raise BaselineIntegrityError('saved diagnostic identity mismatch')
        report=previous
        if report['status']=='COMPLETE': publish(root,output,report);return
        if report['status']!='INCOMPLETE': publish(root,output,report);return
    start=time.monotonic(); execution_deadline=start+plan['execution_seconds']
    model=None; original_hash=None
    def persist():
        write(report_path,report)
        bundle(output,{'source':source,'protocol':identity['diagnostic_protocol_sha256'],'data':identity['data_manifest_sha256']})
    try:
        report['functional_preflight']=functional_preflight()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        assets=output.with_name(output.name+'-frozen-assets')
        recovery_start=time.monotonic()
        # Interrupt even an outstanding network request when recovery's independent budget expires.
        def timeout_handler(signum,frame): raise TimeoutError('frozen asset recovery exceeded its budget')
        old_handler=signal.signal(signal.SIGALRM,timeout_handler)
        signal.setitimer(signal.ITIMER_REAL,plan['recovery_seconds'])
        try:
            recovered=recover_frozen_assets(root,assets,plan['training_identity'],plan['checkpoint_identity'],
                                  min(execution_deadline,time.monotonic()+plan['recovery_seconds']),plan['max_saved_versions'],
                                  progress_output=output,progress_identity={'source':source,'protocol':identity['diagnostic_protocol_sha256'],'data':identity['data_manifest_sha256']})
            if report_path.exists():
                report=json.loads(report_path.read_text())
                if report['identity']!=identity: raise BaselineIntegrityError('recovered diagnostic identity mismatch')
            if recovered!='DIAGNOSTIC_TERMINAL':
                model,tokenizer,documents=load_frozen_model(plan['training_identity'],plan['checkpoint_identity'],assets,base/'protocol.json',device)
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
            report['timings']['asset_recovery_seconds']=time.monotonic()-recovery_start
        if recovered=='DIAGNOSTIC_TERMINAL':
            raise CompletedDiagnostic()
        report['asset_provenance']=json.loads((assets/'asset-provenance.json').read_text())
        documents=documents[:plan['documents']]
        if len(documents)!=plan['documents']: raise BaselineIntegrityError('fewer than 16 original development documents')
        from dataclasses import asdict
        report['model_config']=asdict(model.config)
        report['data_assets']={name:sha256_file(assets/'assets'/name) for name in ('manifest.json','tokenizer.json','development.json')}
        original_hash=weight_hash(model)
        report['weights_before']=original_hash
        deadline=min(execution_deadline,time.monotonic()+plan['diagnostic_seconds'])
        probes=[]
        def task(name,fn):
            check_deadline(deadline);began=time.monotonic(); result=fn()
            report['timings'][name]=time.monotonic()-began
            return result
        # Baseline is always reconstructed to obtain only the few fixed branch states.
        baseline=task('D0_baseline',lambda:run_documents(model,documents,deadline,probes=probes,fractions=plan['probe_fractions']))
        if abs(baseline['nll']-plan['expected_validation_nll'])>plan['baseline_atol']:
            report['D0']={'status':'FAILED','observed':baseline,'expected_nll':plan['expected_validation_nll']}
            raise BaselineIntegrityError('D0 validation NLL exceeds frozen 1e-4 tolerance')
        checks=integrity_checks(model,documents[0],deadline)
        repeated=task('D0_repeat',lambda:run_documents(model,documents,deadline))
        if baseline['stories']!=repeated['stories']: raise BaselineIntegrityError('full baseline repeat mismatch')
        if weight_hash(model)!=original_hash: raise BaselineIntegrityError('D0 mutated frozen weights')
        report['D0']={'status':'PASS',**baseline,'checks':checks,'expected_nll':plan['expected_validation_nll'],'atol':plan['baseline_atol']}
        persist()
        for section,modes in [('D1',['RESET_EVERY_TOKEN','RESET_AT_PROBE','ENTRY_ONLY']),
                              ('D2',['NEIGHBORS_REPLACED_BY_SELF','CROSS_CELL_REMOVED'])]:
            values=report.setdefault(section,{})
            for mode in modes:
                if mode not in values:
                    result=task(mode,lambda:run_documents(model,documents,deadline,mode=mode))
                    result['comparison']=comparison(result,baseline,plan)
                    if mode=='RESET_AT_PROBE':
                        def tail(value):
                            return {'stories':[{'story_id':r['story_id'],'tokens':r['probe_tail_tokens'],
                                                'nll_sum':r['probe_tail_sum']} for r in value['stories']]}
                        result['post_probe_comparison']=comparison(tail(result),tail(baseline),plan)
                    if mode=='ENTRY_ONLY': result['intervention_schedule']='clear non-entry cells before every event'
                    values[mode]=result;persist()
        if 'history_pairs' not in report['D1']:
            report['D1']['history_pairs']=task('D1_history_pairs',lambda:history_pairs(model,tokenizer,plan,deadline));persist()
        values=report.setdefault('D3',{});values['4']=baseline
        for k in (2,8):
            if str(k) not in values:
                result=task('D3_K'+str(k),lambda:run_documents(model,documents,deadline,steps=k))
                result['comparison']=comparison(result,baseline,plan);values[str(k)]=result;persist()
        if 'event_probes' not in report:
            communication,evolution=task('event_probes',lambda:event_probes(model,probes,plan,deadline))
            write(output/'event-probes-private.json',{'connection':communication,'evolution':evolution})
            report['D2']['single_event_summary']=probe_summary(communication['single_event'],'delta_nll')
            report['D2']['cell_sensitivity_summary']={str(cell):probe_summary([r for r in communication['cell_sensitivity'] if r['cell']==cell],'delta_nll') for cell in plan['probe_cells']}
            report['D3']['extra_evolution_summary']={str(extra):probe_summary([r for r in evolution['single_event_extra_evolution'] if r['extra_steps']==extra],'gain') for extra in (0,*plan['extra_evolve'])}
            report['event_probes']={'documents':len(documents),'positions_per_document':plan['probe_fractions'],'events':len(probes),'normal_history_K':4};persist()
        findings=[]
        if report['D1']['RESET_AT_PROBE']['comparison']['document_bootstrap_95'][0]>0: findings.append('STATE_DEPENDENCE_OBSERVED')
        history=report['D1']['history_pairs']
        if history['status']=='COMPLETE' and history.get('bidirectional_pair_success_fraction',0)>=.75 and history['means']['normal']['correct_target_nll']<history['means']['reset']['correct_target_nll']:
            findings.append('SEMANTIC_HISTORY_SIGNAL_OBSERVED')
        if report['D2']['CROSS_CELL_REMOVED']['comparison']['document_bootstrap_95'][0]>0: findings.append('COMMUNICATION_DEPENDENCE_OBSERVED')
        if any(r.get('comparison',{}).get('document_bootstrap_95',[0,0])[1]<0 for k,r in values.items() if k=='8'):
            findings.append('INTERNAL_EVOLUTION_GAIN_OBSERVED')
        report['finding_rules']='exploratory labels: natural-document paired interval excludes zero in expected direction; semantic pair rule recorded separately; no confirmation'
        report['findings']=findings or ['NO_CLEAR_SIGNAL']
        if history['status']=='INCONCLUSIVE': report['findings'].append('INCONCLUSIVE')
        report.pop('error',None)
        report['status']='COMPLETE'
    except CompletedDiagnostic:
        pass
    except AssetUnavailable as error:
        report.update(status='BLOCKED_MODEL_ASSET_UNAVAILABLE',error=str(error))
    except BaselineIntegrityError as error:
        report.update(status='FAILED_BASELINE_INTEGRITY',error=str(error))
    except (FloatingPointError,RuntimeError) as error:
        numerical=isinstance(error,FloatingPointError) or 'assert' in str(error).lower() or 'non-finite' in str(error).lower()
        report.update(status='FAILED_NUMERICAL_CHECK' if numerical else 'BLOCKED_MODEL_ASSET_UNAVAILABLE',error=str(error))
    except TimeoutError as error:
        report.update(status='INCOMPLETE' if model is not None else 'BLOCKED_MODEL_ASSET_UNAVAILABLE',error=str(error))
    except (OSError,ValueError,KeyError) as error:
        report.update(status='FAILED_BASELINE_INTEGRITY' if model is not None else 'BLOCKED_MODEL_ASSET_UNAVAILABLE',error=str(error))
    finally:
        # A recovery/network failure may occur after a partial progress package was
        # restored but before the loader returned. Preserve those completed sections.
        if model is None and report_path.exists():
            prior=json.loads(report_path.read_text())
            if prior.get('identity')==identity:
                for key in ('D0','D1','D2','D3','event_probes'):
                    if key in prior: report.setdefault(key,prior[key])
        if model is not None and original_hash is not None:
            try:
                final_hash=weight_hash(model);report['weights_after']=final_hash
                if final_hash!=original_hash: report.update(status='FAILED_BASELINE_INTEGRITY',error='frozen model weights changed')
            except RuntimeError as error: report.update(status='FAILED_NUMERICAL_CHECK',error=str(error))
        report['timings']['execution_seconds']=time.monotonic()-start
        persist()
    publish(root,output,report)
    print(json.dumps({'mode':'FROZEN_MODEL_DIAGNOSTIC','status':report['status'],'report':str(report_path)}))
