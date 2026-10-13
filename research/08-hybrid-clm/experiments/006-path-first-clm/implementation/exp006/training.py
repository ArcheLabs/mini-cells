"""Whole hard-trajectory score-function training; no backup cells or local edge probes."""
from __future__ import annotations
import json,math,platform,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .baseline import make_model
from .accounting import ComputeCounter
from .evaluation import evaluate,diagnostics,intervention_suite
from .checkpoint import save,restore,file_sha256
from .reuse import make_batch,write_json

def objective(out,y,c):
    per_token=F.cross_entropy(out.logits.float().flatten(0,1),y.flatten(),reduction='none').reshape(y.shape)
    if out.log_prob.numel()==0:return per_token.mean()+c.get('moe_balance_coefficient',0.01)*out.balance,{'ce':float(per_token.mean().detach()),'actor':0.,'critic':0.,'entropy':0.,'moe_balance':float(out.balance.detach())}
    # J is a terminal per-token loss: sum log-probs over decisions; normalize by tokens only.
    advantage=per_token.detach()[...,None]-out.values.detach()
    actor=(advantage*out.log_prob).sum(-1).mean()
    critic=(out.values-per_token.detach()[...,None]).square().mean()
    entropy=out.entropy.mean()
    loss=per_token.mean()+actor+c['critic_coefficient']*critic-c['entropy_coefficient']*entropy
    return loss,{'ce':float(per_token.mean().detach()),'actor':float(actor.detach()),'critic':float(critic.detach()),'entropy':float(entropy.detach()),
                 'terminal_nll_std':float(per_token.detach().std(unbiased=False)),
                 'advantage_std':float(advantage.std(unbiased=False)),
                 'score_term_std':float((advantage*out.log_prob.detach()).sum(-1).std(unbiased=False))}

def gradient_diagnostics(model,before):
    groups={'controller':[],'cells':[],'memory':[],'shared':[]}
    for name,p in model.named_parameters():
        label='controller' if 'controller' in name else 'cells' if 'cells' in name or 'ffn' in name else 'memory' if 'memory' in name or 'context' in name else 'shared'
        groups[label].append((name,p))
    report={}
    for k,rows in groups.items():
        grads=[p.grad.detach().float() for _,p in rows if p.grad is not None]
        report[k]={'gradient_norm':math.sqrt(sum(float(g.square().sum()) for g in grads)),
                   'nonfinite_gradient_entries':sum(int((~torch.isfinite(g)).sum()) for g in grads),
                   'update_norm':math.sqrt(sum(float((p.detach()-before[n]).float().square().sum()) for n,p in rows))}
    return report

def _train(c,run_id,assets,output,device='cuda:0',resume=True):
    dest=Path(output);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'metrics.json').exists():raise FileExistsError('completed result is immutable')
    seed=run_id['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=make_model(c['model'],run_id['arm'],seed).to(device);opt=torch.optim.AdamW(model.parameters(),lr=c['training']['learning_rate'],betas=(.9,.95),weight_decay=c['training']['weight_decay'])
    data_rng=np.random.default_rng(seed);path_rng=torch.Generator(device=device).manual_seed(seed+1)
    latest=dest/'latest.pt';step=0;history=[];counter=ComputeCounter(model);tokens=c['training']['batch_size']*c['training']['sequence_length'];started=time.perf_counter()
    if resume and latest.exists():
        step,extra=restore(latest,model,opt,run_id,data_rng,path_rng);history=extra['history'];counter.by_module.update(extra['compute'])
    try:
        model.train()
        with counter:
            counter.enabled=False
            for step in range(step+1,c['training']['steps']+1):
                starts=data_rng.choice(assets['train_starts'],size=c['training']['batch_size'],replace=True)
                x,y=make_batch(assets['train'],starts,c['training']['sequence_length'],torch.device(device))
                schedule=c['training'];warm=min(1.,step/max(1,schedule['warmup_steps']));progress=max(0,(step-schedule['warmup_steps'])/max(1,schedule['steps']-schedule['warmup_steps']))
                for group in opt.param_groups:group['lr']=schedule['learning_rate']*warm*(.1+.9*.5*(1+math.cos(math.pi*progress)))
                opt.zero_grad(set_to_none=True);prior_flops=counter.total;counter.enabled=True;out=model(x,policy='sampled',generator=path_rng);counter.enabled=False
                loss,parts=objective(out,y,schedule);parts['path_cost_gemm_per_token']=(counter.total-prior_flops)/tokens
                if not torch.isfinite(loss):raise RuntimeError('nonfinite objective')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('nonfinite gradients')
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                diagnostic=step==1 or step%schedule['evaluate_every']==0
                before={n:p.detach().clone() for n,p in model.named_parameters()} if diagnostic else {}
                opt.step()
                if diagnostic:
                    row={'step':step,'tokens':step*tokens,**parts,'gradients':gradient_diagnostics(model,before),'path':diagnostics(out.actions,c['model']['cell_count']) if model.arm!='M0' else {},'sampled_training_compute':counter.report(step*tokens)}
                    role=run_id['role'];row['greedy_eval']=evaluate(model,assets[role],assets['eval_starts'],assets['eval_groups'],c)
                    row['sampled_eval']=evaluate(model,assets[role],assets['eval_starts'],assets['eval_groups'],c,policy='sampled');history.append(row)
                    write_json(dest/'learning_curve.json',history);print(json.dumps({'arm':model.arm,'step':step,'ce':parts['ce'],'greedy_ppl':row['greedy_eval']['perplexity']}),flush=True)
                if step%schedule['checkpoint_every']==0 or diagnostic or step==schedule['steps']:
                    save(latest,model,opt,step,run_id,data_rng,path_rng,{'history':history,'compute':dict(counter.by_module)})
        role=run_id['role'];greedy=evaluate(model,assets[role],assets['eval_starts'],assets['eval_groups'],c);sampled=evaluate(model,assets[role],assets['eval_starts'],assets['eval_groups'],c,policy='sampled')
        metrics={'identity':run_id,'status':'completed','scientific_decision':False,'model_config':c['model'],'checkpoint_sha':file_sha256(latest),
                 'learning_curve':history,'development_eval':greedy if role=='development' else None,'confirmation_holdout':greedy if role=='confirmation' else None,
                 'sampled_eval':sampled,'routing_diagnostics':greedy['routing'],'sampled_greedy_nll_gap':sampled['nll']-greedy['nll'],
                 'trajectory_distribution_gap':{'greedy':greedy['routing'],'sampled':sampled['routing']},'compute_accounting':{'training':counter.report(step*tokens),'greedy_evaluation':greedy['compute'],'parameter_breakdown':model.parameter_breakdown()},
                 'decode_profile':None,'interventions':intervention_suite(model,assets[role],assets['eval_starts'],assets['eval_groups'],c),
                 'runtime_versions':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'device':str(device),'dtype':str(next(model.parameters()).dtype)},
                 'steps':step,'train_tokens':step*tokens,'elapsed_seconds':time.perf_counter()-started,'sample_generation':generate(model,x[:1,:min(8,x.shape[1])],8)}
        write_json(dest/'metrics.json',metrics);return metrics
    except BaseException as exc:
        write_json(dest/'failure.json',{'identity':run_id,'status':'INCOMPLETE','scientific_decision':False,'phase':'train_or_evaluate','step':step,'error_type':type(exc).__name__,'reason':str(exc),'last_checkpoint_step':torch.load(latest,map_location='cpu',weights_only=False)['step'] if latest.exists() else None});raise

@torch.inference_mode()
def generate(model,prefix,count):
    model.eval();out=model.prefill(prefix);rows=[]
    for _ in range(min(count,model.config['max_sequence_length']-prefix.shape[1])):
        token=out.logits[:,-1].argmax(-1);rows.append(token.tolist());out=model.decode_step(token,out.memory)
    return {'greedy_token_ids':rows,'tokenizer_decode':'use bundle tokenizer; sample is diagnostic only'}


def train(c,run_id,assets,output,device='cuda:0',resume=True):
    """Public training API retains the gate even when called outside the CLI."""
    from .config import load,authorize
    from .checkpoint import identity
    _,protocol,_=load()
    authorize(c,protocol,run_id['seed'],run_id['role'])
    expected=identity(run_id['source_commit'],c,protocol,assets['manifest_sha256'],
                      run_id['arm'],run_id['seed'],run_id['role'])
    if expected!=run_id:raise ValueError('training identity does not match approved contract')
    return _train(c,run_id,assets,output,device,resume)
