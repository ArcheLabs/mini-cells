"""Separate randomized token-ID mechanism tasks; not TinyStories evidence."""
import random
import torch
from torch.nn import functional as F
from runtime import Session
from common import model_hash,sha,native


def episodes(model,seed,per_task,exclude=()):
    rng=random.Random(seed);used=set(exclude);result=[]
    reach=sum(s.iterations*(s.window-1) for s in model.stages)
    gap=32*((reach+31)//32+2)
    for task in ('M1','M2','M3','M4'):
        for _ in range(per_task):
            while True:
                keys=rng.sample(range(32,48),4);values=rng.sample(range(64,80),4)
                signature=(task,*keys,*values)
                if signature not in used:break
            used.add(signature);tokens=[2,13 if task=='M4' else 10,keys[0],values[0]]
            tokens.extend(rng.randrange(100,128) for _ in range(gap+64-len(tokens)))
            target=values[0]
            if task=='M2':tokens[32:35]=[11,keys[0],values[1]];target=values[1]
            if task=='M3':
                for position in range(32,len(tokens)-16,32):tokens[position:position+3]=[10,keys[1],values[(position//32)%3+1]]
            if task=='M4':tokens[-8:-4]=rng.sample(values,4)
            tokens.extend([12,keys[0]])
            result.append({'task':task,'inputs':tokens,'target':target,'combination':signature,'receptive_field_bound':reach,'early_fact_gap':gap})
    return result,used


def evaluate(model,examples,device,deadline):
    import time
    model.eval();rows=[]
    with torch.no_grad():
        for example in examples:
            if time.monotonic()>=deadline:raise TimeoutError('mechanism evaluation budget exhausted')
            ids=torch.tensor([example['inputs']],device=device)
            logits=model.forward_extended(ids)[:,-1];prediction=int(logits.argmax(-1))
            rows.append({'task':example['task'],'combination':list(example['combination']),'correct':prediction==example['target'],
                         'prediction':prediction,'target':example['target'],'input_tokens':ids.numel()})
    return {'status':'COMPLETE','tasks':{task:{'accuracy':sum(r['correct'] for r in rows if r['task']==task)/sum(r['task']==task for r in rows),
             'examples':sum(r['task']==task for r in rows)} for task in ('M1','M2','M3','M4')},'examples':rows,
             'interpretation':'random token-ID mechanism accuracy; does not establish natural-language long-horizon reasoning'}


def run(model,plan,device,deadline,output=None):
    import copy,time
    training,used=episodes(model,plan['train_seed'],plan['train_combinations_per_task'])
    testing,_=episodes(model,plan['evaluation_seed'],plan['evaluation_combinations_per_task'],used)
    candidate=copy.deepcopy(model).train();opt=plan['optimizer']
    optimizer=torch.optim.AdamW(candidate.parameters(),lr=opt['lr'],betas=tuple(opt['betas']),weight_decay=opt['weight_decay'])
    targets=observed=0;losses=[]
    for example in training:
        if time.monotonic()>=deadline:raise TimeoutError('mechanism task-training budget exhausted')
        optimizer.zero_grad(set_to_none=True);ids=torch.tensor([example['inputs']],device=device)
        logits=candidate.forward_extended(ids)[:,-1];loss=F.cross_entropy(logits,torch.tensor([example['target']],device=device))
        if not bool(loss.isfinite()):raise FloatingPointError('nonfinite mechanism loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(candidate.parameters(),opt['gradient_clip'],error_if_nonfinite=True);optimizer.step()
        targets+=1;observed+=ids.numel();losses.append(float(loss.detach()))
    if output is not None:
        output.mkdir(parents=True,exist_ok=True)
        native._atomic_save({'configuration':candidate.configuration,'model':candidate.state_dict(),'language_model_weight_identity':model_hash(model),'task_training_input_sha256':sha(training)},output/'mechanism-final-model.pt')
    result=evaluate(candidate,testing,device,deadline);result.update(extra_training_prediction_targets=targets,
        extra_training_input_tokens=observed,losses=losses,training_input_sha256=sha(training),evaluation_input_sha256=sha(testing),training_examples=training,evaluation_inputs=testing,task_weights='private/mechanism-final-model.pt' if output is not None else None,task_trained_weight_identity=model_hash(candidate),
        language_model_weight_identity=model_hash(model),separate_training=True,tbptt_chunk_counts=sorted({(len(r['inputs'])+31)//32 for r in training}),tbptt='entire delayed episode; no detach before query loss')
    if candidate.persistent:
        try:
            with torch.no_grad():
                ids=torch.tensor([testing[0]['inputs']],device=device);session=Session(candidate.eval(),'x4c-mechanism-token-id-schema-v1')
                split=39;session.prefill(ids[:,:split]);saved=session.snapshot();a=session.prefill(ids[:,split:])
                b=Session(candidate,'x4c-mechanism-token-id-schema-v1').restore(saved).prefill(ids[:,split:])
                result['M5']={'status':'PASS' if torch.equal(a,b) else 'FAIL','no_history_replay':True}
        except Exception as error:result['M5']={'status':'FAILED','error':type(error).__name__+': '+str(error)}
        if model.configuration['name'] in ('B2','C2'):
            for key,read,write in (('read_disabled_frozen',False,True),('write_disabled_frozen',True,False)):
                try:
                    candidate.read_enabled=read;candidate.write_enabled=write
                    result[key]=evaluate(candidate,testing,device,deadline)
                except Exception as error:result[key]={'status':'FAILED','error':type(error).__name__+': '+str(error)}
                finally:candidate.read_enabled=True;candidate.write_enabled=True
            if model.configuration['name']=='B2':
                result['B3_read_disabled']=result['read_disabled_frozen'];result['B4_write_disabled']=result['write_disabled_frozen']
        if result['M5']['status']!='PASS' or any(result[key]['status']!='COMPLETE' for key in ('read_disabled_frozen','write_disabled_frozen') if key in result):result['status']='INCOMPLETE'
    else:result['M5']={'status':'NOT_APPLICABLE_NO_PERSISTENT_STATE'}
    return result
