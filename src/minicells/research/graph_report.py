"""004 partial-session reports and strict paired, three-seed development decision."""
import argparse
import json
import math
from pathlib import Path
from minicells.research.config import dump, digest, ROOT

MODELS = ['fixed-cell', 'global-router', 'soft-cell', 'hard-cell', 'graph-cell']


def summarize_session(output, config):
    output = Path(output)
    rows, routes, logs = [], {}, []
    for kind in config['run_models']:
        directory = output / kind
        metrics = directory / 'metrics.json'
        rows.append(json.loads(metrics.read_text()) if metrics.exists() else {'model':kind,'status':'missing'})
        route = directory / 'routing_stats.json'
        if route.exists():
            routes[kind] = json.loads(route.read_text())
        log = directory / 'logs.jsonl'
        if log.exists():
            logs.extend(dict(json.loads(line), model=kind) for line in log.read_text().splitlines() if line.strip())
    complete = all(row['status'] == 'completed' and row.get('train_tokens') == config['train_tokens'] for row in rows)
    decision = {'status':'DEVELOPMENT_SESSION_COMPLETED' if complete else 'DEVELOPMENT_INCOMPLETE',
                'seed':config['seed'],'session':config['session'],'scientific_decision':False,
                'gates':None,'note':'Both sessions for all three seeds are required; a completed session is not matrix support.'}
    dump(output / 'metrics.json', {'models':rows,'status':decision['status']})
    dump(output / 'routing_stats.json',routes)
    (output / 'logs.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in logs))
    dump(output / 'decision.json',decision)
    return decision


def visualize(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from minicells.research.report import generate
    generate(output)
    output = Path(output)
    routes = json.loads((output / 'routing_stats.json').read_text())
    graph = routes.get('graph-cell',{}).get('layers',[])
    for name in ('graph_topology','path_heatmap','transition_entropy'):
        fig, ax = plt.subplots(figsize=(9,6))
        if not graph:
            ax.text(.5,.5,'Graph measurement unavailable in this session',ha='center',transform=ax.transAxes)
        elif name == 'graph_topology':
            # All layers represented separately; a selected example is never a gate.
            fig.clf()
            for ordinal, row in enumerate(graph):
                ax = fig.add_subplot(2, math.ceil(len(graph)/2), ordinal+1)
                prior=np.asarray(row['graph_prior_probabilities'])
                nodes=prior.shape[1]; angles=np.arange(nodes)*2*np.pi/nodes
                xy=np.stack([np.cos(angles),np.sin(angles)],axis=1)
                for i,j in zip(*np.where(prior >= .05)):
                    if i != j:
                        ax.annotate('',xy=xy[j],xytext=xy[i],arrowprops={'arrowstyle':'->','alpha':float(min(.85,prior[i,j]+.15))})
                for i,point in enumerate(xy):
                    ax.text(*point, 'STOP' if i==nodes-1 else str(i),ha='center',va='center',bbox={'boxstyle':'circle','fc':'white'})
                ax.set(xlim=(-1.4,1.4),ylim=(-1.4,1.4),title='Layer '+str(row['layer']))
                ax.axis('off')
            fig.suptitle('Persistent prior edges p >= 0.05 (contextual transitions differ; self-loops omitted)')
        elif name == 'path_heatmap':
            values=[];labels=[]
            for row in graph:
                top=sorted(row.get('path_counts',[]),key=lambda item:item['count'],reverse=True)[:8]
                for item in top:
                    path=item['path'];values.append([int(cell) if isinstance(cell,int) else -1 if cell=='STOP' else -2 for cell in path]+[-3]*(5-len(path)))
                    labels.append('L{} {} tokens'.format(row['layer'],item['count']))
            if values:
                picture=ax.imshow(values,aspect='auto',vmin=-3,vmax=7)
                ax.set_yticks(range(len(labels)));ax.set_yticklabels(labels,fontsize=6)
                ax.set(xlabel='Hop / terminal',title='Top eight paths per layer; -1 STOP, -2 forced, -3 padding')
                fig.colorbar(picture,ax=ax,label='Cell / terminal code')
        else:
            for row in graph:
                ax.plot(row['graph_prior_entropy_nats'],marker='.',label='L'+str(row['layer'])+' prior')
                ax.plot(row['empirical_transition_entropy_nats'],linestyle='--',label='L'+str(row['layer'])+' observed')
            ax.set(xlabel='Source Cell',ylabel='Transition entropy (nats)');ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(output/'plots'/(name+'.png'),dpi=160);plt.close(fig)


def paired_decision(reference, graph, config, protocol):
    reference, graph = Path(reference), Path(graph)
    reasons=[]
    identities=[]
    for directory, expected in ((reference,'reference'),(graph,'graph')):
        manifest_path=directory/'evidence-manifest.json'
        if not manifest_path.exists():
            reasons.append('missing manifest '+expected);continue
        manifest=json.loads(manifest_path.read_text())
        for required in ('metrics.json','routing_stats.json','config.json','experiment.json','decision.json','logs.jsonl'):
            if required not in manifest.get('files',{}):
                reasons.append('manifest missing '+expected+'/'+required)
        for name, sha in manifest.get('files',{}).items():
            file=(directory/name).resolve()
            if not file.is_relative_to(directory.resolve()) or not file.is_file() or digest(file)!=sha:
                reasons.append('invalid checksum '+expected+'/'+name)
        identities.append(manifest)
        if (directory/'experiment.json').exists():
            manifest['environment'] = {key:json.loads((directory/'experiment.json').read_text()).get(key) for key in ('torch','python','dependencies','attention_backend')}
        if manifest.get('source_commit') is None or manifest.get('assets_manifest_sha256') is None:
            reasons.append('source/input identity missing')
        if manifest.get('session')!=expected or manifest.get('seed')!=config['seed'] or manifest.get('scientific_decision') is not False:
            reasons.append('session/seed identity differs '+expected)
        if manifest.get('config_sha256') != digest(ROOT/'research/projects/cellular-language-models/experiments/004-cell-graph/config/registered.yaml') or manifest.get('protocol_sha256') != digest(ROOT/'research/projects/cellular-language-models/experiments/004-cell-graph/protocol.json'):
            reasons.append('config/protocol not registered')
    if len(identities)==2:
        for field in ('source_commit','definition_sha256','config_sha256','protocol_sha256','assets_manifest_sha256','seed','environment'):
            if identities[0].get(field)!=identities[1].get(field):
                reasons.append('paired identity mismatch '+field)
    models={};routes={}
    for directory in (reference,graph):
        path=directory/'metrics.json'
        if path.exists():
            for row in json.loads(path.read_text()).get('models',[]):
                if row['model'] in models: reasons.append('duplicate arm '+row['model'])
                models[row['model']]=row
        path=directory/'routing_stats.json'
        if path.exists():routes.update(json.loads(path.read_text()))
    for kind in config['models']:
        row=models.get(kind,{})
        ev=row.get('evaluation',{})
        if row.get('status')!='completed' or row.get('train_tokens')!=config['train_tokens'] or row.get('seed')!=config['seed']:
            reasons.append('full-budget arm missing '+kind)
        if ev.get('validation_tokens')!=config['validation_batches']*config['batch_size']*config['sequence_length']:
            reasons.append('validation incomplete '+kind)
        if ev.get('uncovered_attention_calls') or not ev.get('observed_gemm_calls'):
            reasons.append('uncovered or unobserved attention/matmul '+kind)
        for key in ('perplexity','executed_gemm_flops_per_token','inference_latency_seconds_per_batch'):
            if not isinstance(ev.get(key),(int,float)) or not math.isfinite(ev[key]) or ev[key]<=0:
                reasons.append('measurement missing '+kind+'/'+key)
    capacities=[row.get('params',0) for row in models.values()]
    if len(capacities)!=5 or not min(capacities) or max(capacities)/min(capacities)-1>protocol['capacity_tolerance_fraction']:
        reasons.append('capacity mismatch')
    hard_layers=routes.get('hard-cell',{}).get('layers',[]);graph_layers=routes.get('graph-cell',{}).get('layers',[])
    if len(hard_layers)!=config['layers'] or len(graph_layers)!=config['layers']:
        reasons.append('routing layers incomplete')
    else:
        for row in hard_layers+graph_layers:
            if row.get('path_population')!=config['validation_batches']*config['batch_size']*config['sequence_length'] or row.get('path_entropy_nats') is None:
                reasons.append('full path population missing')
    if reasons:return {'seed':config['seed'],'status':'DEVELOPMENT_INCOMPLETE','scientific_decision':False,'gates':None,'reasons':reasons}
    graph_entropy=sum(row['path_entropy_nats'] for row in graph_layers)/len(graph_layers)
    hard_entropy=sum(row['path_entropy_nats'] for row in hard_layers)/len(hard_layers)
    graph_ev=models['graph-cell']['evaluation'];fixed_ev=models['fixed-cell']['evaluation']
    gates={'A_path_entropy':graph_entropy<hard_entropy,'B_ppl':graph_ev['perplexity']<=13,
           'C_gemm_reduction':graph_ev['executed_gemm_flops_per_token']<.5*fixed_ev['executed_gemm_flops_per_token'],
           'anti_collapse':all(row['path_diversity']>=2 and sum(n>0 for n in row['non_entry_cell_counts'].values())>=2 for row in graph_layers)}
    return {'seed':config['seed'],'status':'DEVELOPMENT_SUPPORTED' if all(gates.values()) else 'DEVELOPMENT_NOT_SUPPORTED',
            'scientific_decision':False,'gates':gates,'source_commit':identities[0]['source_commit'],
            'assets_manifest_sha256':identities[0]['assets_manifest_sha256'],'environment':identities[0].get('environment'),
            'comparison':{'graph_path_entropy':graph_entropy,'hard_path_entropy':hard_entropy,'graph_perplexity':graph_ev['perplexity'],
                          'gemm_reduction_vs_fixed':1-graph_ev['executed_gemm_flops_per_token']/fixed_ev['executed_gemm_flops_per_token']}}


def analyze(archive, output):
    config=json.loads((ROOT/'research/projects/cellular-language-models/experiments/004-cell-graph/config/registered.yaml').read_text())
    protocol=json.loads((ROOT/'research/projects/cellular-language-models/experiments/004-cell-graph/protocol.json').read_text())
    archive=Path(archive);output=Path(output)
    results=[]
    measurements=[]
    for seed in config['seeds']:
        for session in ('reference','graph'):
            path=archive/('seed-'+str(seed))/session/'metrics.json'
            if path.exists():
                for row in json.loads(path.read_text()).get('models',[]):
                    measurements.append(dict(row,seed=seed))
        results.append(paired_decision(archive/('seed-'+str(seed))/'reference',archive/('seed-'+str(seed))/'graph',dict(config,seed=seed),protocol))
    identities={(row.get('source_commit'),row.get('assets_manifest_sha256'),json.dumps(row.get('environment'),sort_keys=True)) for row in results}
    complete=all(row['status']!='DEVELOPMENT_INCOMPLETE' for row in results) and len(identities)==1
    status='DEVELOPMENT_INCOMPLETE' if not complete else 'DEVELOPMENT_SUPPORTED' if all(row['status']=='DEVELOPMENT_SUPPORTED' for row in results) else 'DEVELOPMENT_NOT_SUPPORTED'
    import statistics
    valid_seeds={row['seed'] for row in results if row['status']!='DEVELOPMENT_INCOMPLETE'}
    statistics_by_model={}
    for kind in MODELS:
        selected=[row for row in measurements if row['model']==kind and row.get('status')=='completed' and row['seed'] in valid_seeds]
        statistics_by_model[kind]={}
        for key in ('perplexity','executed_gemm_flops_per_token','inference_latency_seconds_per_batch'):
            values=[row['evaluation'][key] for row in selected if key in row.get('evaluation',{})]
            statistics_by_model[kind][key]={'n':len(values),'mean':statistics.mean(values) if values else None,
                                           'sample_sd':statistics.stdev(values) if len(values)>1 else None}
    decision={'status':status,'scientific_decision':False,'seeds':results,'boundary':protocol['claim_boundary'],'descriptive_statistics':statistics_by_model}
    dump(output/'decision.json',decision)
    (output/'analysis.md').write_text('# Experiment 004 matrix\n\n'+status+'\n\n```json\n'+json.dumps(decision,indent=2)+'\n```\n\n'+protocol['claim_boundary']+'\n')
    return decision


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--archive',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(analyze(args.archive,args.output)))
