"""Historical Native A2 data order/optimizer/schedule; fresh arms and explicit State gradients."""
import random
import time
import torch
from torch.nn import functional as F
from bootstrap import shared,old_evaluate


def gradient_metrics(model,scale):
    def norm(parameters):
        values=[p.grad.detach().float().norm() for p in parameters if p.grad is not None]
        return float(torch.stack(values).norm()) if values else 0.
    paths={}
    for name,values in model.path_tensors.items():
        gradients=[x.grad.detach().float().norm()/scale for x in values if x.grad is not None]
        paths[name]=float(torch.stack(gradients).norm()) if gradients else 0.
    return {'state_update_gradient_norm':norm(model.persistent.update.parameters()),
            'state_read_projection_gradient_norm':norm(model.persistent.bridge.parameters()),
            'state_write_input_weight_gradient_norm':norm([model.persistent.update.weight_ih]),
            'state_write_recurrent_weight_gradient_norm':norm([model.persistent.update.weight_hh]),
            'initial_slots_gradient_norm':norm([model.persistent.slot]),'read_gate_gradient_norm':norm([model.persistent.gate]),
            'interaction_path_gradient_norms':paths,
            'shared_qkv_gradient_norm_total':norm(p for s in model.stages for p in s.attn.qkv.parameters())}


def run(model,corpus,starts,plan,device,out,deadline,entry):
    profile=plan['profiles']['dev'];target=profile['tokens'];steps=target//1000
    if (target,steps,plan['batch'],plan['sequence_length'])!=(1000000,1000,8,125) and not entry.get('synthetic_fixture'):raise ValueError('historical A2 training budget/config mismatch')
    train=shared.native.open_memmap(corpus.train_path);validation=shared.native.open_memmap(corpus.validation_path)
    optimizer=torch.optim.AdamW(model.parameters(),lr=plan['lr'],betas=tuple(plan['betas']),weight_decay=plan['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda',init_scale=plan['initial_scale'])
    rng=random.Random(91001);curve=[];overflows=[];tokens=updates=0;began=time.monotonic()
    entry['training_status']='IN_PROGRESS';model.collect_gradients=True
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    try:
        for step in range(1,steps+1):
            if time.monotonic()>=deadline:raise TimeoutError('training budget exhausted; incomplete arm restarts fresh')
            locations=tuple(rng.randrange(len(train)-126) for _ in range(8));ids,targets=shared.native.memmap_batch(train,locations,125,device)
            lr=plan['lr']*shared.native.lr_multiplier(step,steps,profile['warmup'])
            for group in optimizer.param_groups:group['lr']=lr
            start=time.monotonic()
            for attempt in range(plan['maximum_consecutive_overflows']):
                model.train();optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                    logits=model(ids);loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1))
                if not bool(loss.isfinite()):raise FloatingPointError('nonfinite loss')
                scale=scaler.get_scale();scaler.scale(loss).backward();scaler.unscale_(optimizer)
                finite=bool(torch.stack([p.grad.isfinite().all() for p in model.parameters() if p.grad is not None]).all())
                if finite:break
                overflows.append({'update':step,'attempt':attempt+1,'committed_tokens':tokens,'loss':float(loss.detach()),'scale':scale,'input_offsets':locations})
                if not scaler.is_enabled():raise FloatingPointError('nonfinite FP32 gradients; no optimizer update')
                scaler.step(optimizer);scaler.update()
                if attempt+1==plan['maximum_consecutive_overflows']:raise FloatingPointError('FP16 overflow tolerance exceeded')
            gradients=gradient_metrics(model,scale)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),plan['clip'],error_if_nonfinite=True)
            scaler.step(optimizer);scaler.update()
            if not bool(torch.stack([p.isfinite().all() for p in model.parameters()]).all()):raise FloatingPointError('nonfinite parameters')
            tokens+=targets.numel();updates+=1;old_evaluate.sync(device)
            curve.append({'update':updates,'prediction_tokens':tokens,'input_offsets':locations,'loss':float(loss.detach()),
                          'learning_rate':lr,'gradient_norm':float(norm),'last_chunk_state_update':{k:float(v) for k,v in model.persistent.last.items()},'seconds':time.monotonic()-start,**gradients})
            model.path_tensors={'attention':[],'attention_scores':[],'ffn':[]}
            if tokens in (100000,250000,500000,750000,target):
                try:curve[-1]['validation']=old_evaluate.language(model,validation,starts,device,deadline)
                except Exception as error:curve[-1]['validation']={'status':'FAILED','error':str(error)}
                shared.write(out/'training-metrics.json',{'metrics':curve,'overflows':overflows})
                print({'arm':model.arm,'tokens':tokens,'loss':curve[-1]['loss']},flush=True)
    finally:
        model.collect_gradients=False;model.path_tensors={'attention':[],'attention_scores':[],'ffn':[]}
        entry.update(training_status='COMPLETE' if tokens==target and updates==steps else 'INCOMPLETE',training_tokens=tokens,
                     optimizer_updates=updates,training_seconds=time.monotonic()-began,
                     training_peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                     training_peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None)
        shared.write(out/'training-metrics.json',{'status':entry['training_status'],'metrics':curve,'overflows':overflows,
            'training_seconds':entry['training_seconds'],'tokens_per_second':tokens/max(entry['training_seconds'],1e-12),
            'tbptt':'four Chunks within independent 125-token sample; no per-Chunk detach; last 29-token Chunk flushed after logits',
            'gradient_scope':'parameters unscaled before global clipping; retained path gradients divided by the current GradScaler scale'})
