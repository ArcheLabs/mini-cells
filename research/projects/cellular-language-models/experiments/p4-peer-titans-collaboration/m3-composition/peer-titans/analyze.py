"""Stage 3 paired episode analysis and reproducible evidence export."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
from statistics import mean

import numpy as np


def load(path):
    return json.loads(path.read_text())


def bootstrap(values, rng, count):
    values=np.asarray(values,dtype=float)
    draws=rng.integers(len(values),size=(count,len(values)))
    return dict(mean=float(values.mean()),ci95=np.quantile(values[draws].mean(1),[.025,.975]).tolist())


def calculate(rows, config):
    rng=np.random.default_rng(config['bootstrap_seed'])
    count=config['bootstrap_replicates']
    result={}
    for n in config['sizes']:
        for width in config['union_widths']:
            group=[r for r in rows if (r['n'],r['width'])==(n,width)]
            assert len(group)==20
            outcomes={}
            for arm in ('P0','P1','P2','P3','P4'):
                metrics=[r['metrics'][arm] for r in group]
                outcomes[arm]={key:bootstrap([m[key] for m in metrics],rng,count)
                               for key in ('joint',)}
                for key in ('recall','early','late','mse','cosine'):
                    outcomes[arm][key]=[bootstrap([m[key][i] for m in metrics],rng,count) for i in (0,1)]
            deltas={r['episode']:r['metrics']['P2']['joint']-r['metrics']['P0']['joint'] for r in group}
            paired={}
            for key in ('recall','early','late','mse','cosine'):
                paired[key]=[bootstrap([r['metrics']['P2'][key][i]-r['metrics']['P0'][key][i]
                                        for r in group],rng,count) for i in (0,1)]
            paired['acquisition_seconds']=bootstrap([
                r['acquisition']['a']['seconds']+r['acquisition']['b']['seconds']-r['acquisition']['p0']['seconds']
                for r in group],rng,count)
            paired['dynamic_bytes']=bootstrap([
                sum(r['cells'][k]['fast_weights']+r['cells'][k]['momentum'] for k in ('a','b'))-
                r['cells']['p0']['fast_weights']-r['cells']['p0']['momentum'] for r in group],rng,count)
            by_seed={}
            for seed in config['seed_counts']:
                subset=[r for r in group if r['cohort']==int(seed)]
                assert len(subset)==config['seed_counts'][seed]
                by_seed[seed]=dict(episodes=len(subset),p0=mean(r['metrics']['P0']['joint'] for r in subset),
                                   p2=mean(r['metrics']['P2']['joint'] for r in subset),
                                   delta=mean(deltas[r['episode']] for r in subset))

            def acceptance(read):
                fields=[]
                for i in (0,1):
                    pairs=[(read(r)['accepted'][i],read(r)['queries']) for r in group]
                    fields.append(dict(numerator=sum(a for a,_ in pairs),denominator=sum(b for _,b in pairs),
                                       **bootstrap([a/b for a,b in pairs],rng,count)))
                return fields

            result[f'S{n}-'+('A' if width==64 else 'B')]=dict(
                paired_joint_delta=bootstrap(list(deltas.values()),rng,count),
                per_episode_delta=deltas,per_seed=by_seed,arms=outcomes,paired_secondary=paired,
                wrong_domain_acceptance=acceptance(lambda r:r['wrong']),
                unknown_acceptance={arm:acceptance(lambda r:r['unknown'][arm]) for arm in ('P0','P2')},
                known_incorrect_acceptance={arm:acceptance(lambda r:dict(
                    accepted=r['metrics'][arm]['incorrect_accept'],queries=n)) for arm in ('P0','P2')},
                ablations={arm:mean(r['metrics'][arm][i] for r in group)
                           for arm,i in (('P2_A',0),('P2_B',1))},
                resources=dict(union_dynamic_bytes=34048 if width==64 else 67848,peer_dynamic_bytes=68096,
                    union_static_bytes=group[0]['cells']['p0']['static_bytes'],
                    peer_shared_static_bytes=group[0]['cells']['a']['static_bytes'],
                    temporary_batch_state_bytes=group[0]['temp_batch_state_bytes'],
                    union_checkpoint_bytes=mean(r['cells']['p0']['bytes'] for r in group),
                    peer_checkpoint_bytes=mean(sum(r['cells'][k]['bytes'] for k in ('a','b')) for r in group),
                    union_acquisition_seconds=mean(r['acquisition']['p0']['seconds'] for r in group),
                    peer_acquisition_seconds=mean(sum(r['acquisition'][k]['seconds'] for k in ('a','b')) for r in group),
                    **{key:mean(r[key] for r in group) for key in
                       ('batch_seconds','serial_seconds','union_seconds','episode_seconds')},
                    process_peak_rss_bytes=max(r['process_peak_rss_bytes'] for r in group)))
    return result


def render(analysis, manifest, restore):
    lines=['# P4-M3 Stage 3 Peer Titans Screening','',
           '80 paired CPU FP32 episodes across four configurations. The primary comparison is P2 versus P0 joint exact match.',
           '', '| Workload | Union | Peer | Paired delta (95% CI) | Wrong-domain accept A / B |',
           '|---|---:|---:|---:|---:|']
    for key,g in analysis.items():
        d=g['paired_joint_delta']; lo,hi=d['ci95']
        rates=' / '.join(f"{r['numerator']}/{r['denominator']}" for r in g['wrong_domain_acceptance'])
        lines.append(f"| {key} | {g['arms']['P0']['joint']['mean']:.2%} | {g['arms']['P2']['joint']['mean']:.2%} | "
                     f"{d['mean']:+.2%} ({lo:+.2%}, {hi:+.2%}) | {rates} |")
    lines+=['','## Hypotheses','',
            'H1 concerns interference and earlier associations. H2 asks whether joint accuracy benefits from '
            'separate Cells after approximate byte matching. H3 tests each branch with removal and wrong-domain queries. '
            'H4 requires exact batch/serial equivalence. H5 records scaling costs without assuming equal compute.',
            '', '## Protocol and checks','',
            'S8 and S16 use nested associations from the same master mappings. Each configuration has '
            '20 episodes over seeds allocated 7/7/6. Chunk 4 and 128 presentations per association are frozen. '
            'The two acquisition paths together use 122,880 calls and 491,520 presentations. '
            'Both modes use the same episode inputs. Mode A matches each Cell width at 64. Mode B compares a width-129 '
            'union with two width-64 Cells, 248 bytes apart in dynamic state. The memory receives keys and '
            'values during acquisition and queries alone during Recall. Candidate values and labels are used by the scorer.',
            '',
            'P1 isolated reads, P2 batched reads and P3 serial reads match exactly. Branch ablations preserve the retained slot. '
            'Bidirectional acquisition checks and state hashes verify isolation. A different process restored '
            f"all {len(restore['episodes'])*3} checkpoints and reproduced fused and union output hashes.",
            '',
            '## Retention, rejection and uncertainty','',
            '| Workload | Union early / late Recall | Peer early / late Recall | Unknown accept Union / Peer |',
            '|---|---:|---:|---:|']
    for key,g in analysis.items():
        def split(arm):
            return ' / '.join(f"{mean(x['mean'] for x in g['arms'][arm][phase]):.2%}" for phase in ('early','late'))
        u=' / '.join(f"{sum(x['numerator'] for x in g['unknown_acceptance'][arm])}/"
                     f"{sum(x['denominator'] for x in g['unknown_acceptance'][arm])}" for arm in ('P0','P2'))
        lines.append(f'| {key} | {split("P0")} | {split("P2")} | {u} |')
    lines+=['','Early/late groups use first-pass acquisition order and are scored after full replay. '
            'This is a positional retention diagnostic. Wrong-domain and independent unknown queries are distinct. '
            'The frozen acceptance threshold is maximum candidate cosine ≥ 0.8. Exact counts and per-domain '
            'intervals are in analysis.json. Bootstrap resamples complete paired episodes 10,000 times per configuration. '
            'Intervals describe within-screening episode variation at the fixed seeds.',
            '', '## Resource accounting','',
            '| Workload | Dynamic bytes Union / Peer | Acquisition seconds Union / Peer | Read ms Union / batch / serial |',
            '|---|---:|---:|---:|']
    for key,g in analysis.items():
        r=g['resources']
        lines.append(f"| {key} | {r['union_dynamic_bytes']:,} / {r['peer_dynamic_bytes']:,} | "
                     f"{r['union_acquisition_seconds']:.3f} / {r['peer_acquisition_seconds']:.3f} | "
                     f"{r['union_seconds']*1000:.3f} / {r['batch_seconds']*1000:.3f} / {r['serial_seconds']*1000:.3f} |")
    lines+=['','Timings are CPU wall measurements of the existing helpers. The serial helper includes read-state '
            'hash checks. Batch timings include state packing. These measurements do not establish hardware overlap or '
            'equal compute. Peak RSS is the writer process high-water mark, including PyTorch. Full static, checkpoint, '
            'temporary-state and timing data are retained.',
            '', '## Screening decision','',
            'Classify each observed delta using its paired interval. Practical effect size and false-acceptance tolerance '
            'were not pre-registered. Confirmation needs separate review and frozen thresholds.',
            '', f"Run source {manifest['source']}.", '']
    for key,g in analysis.items():
        lo,hi=g['paired_joint_delta']['ci95']
        signal='positive screening signal' if lo>0 else 'negative screening signal' if hi<0 else 'no clear advantage'
        lines.append(f'- {key} {signal}.')
    has_signal=any(g['paired_joint_delta']['ci95'][0]>0 for g in analysis.values())
    if has_signal:
        lines+=['','A positive interval supports reviewing a confirmation design after defining practical '
                 'effect and rejection criteria. The present result qualifies only as screening evidence.']
    else:
        lines+=['','Hold confirmation. The paired screening intervals do not establish a positive '
                 'composition advantage in the tested workload.']
    return '\n'.join(lines)+'\n'


def export(source,destination,report):
    destination.mkdir(parents=True,exist_ok=False)
    manifest=load(source/'manifest.json')
    assert not manifest['calibration']
    restore=load(source/'restore.json')
    assert restore['status']=='PASS' and len(restore['episodes'])==80
    files=sorted((source/'episodes').glob('*.json'))
    rows=[load(f) for f in files]
    assert len(rows)==80 and all(r['status']=='PASS' for r in rows)
    analysis=calculate(rows,manifest['config'])
    (destination/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
    for f in (source/'manifest.json',source/'restore.json'):
        shutil.copy2(f,destination/f.name)
    shutil.copytree(source/'episodes',destination/'episodes')
    receipt={str(f.relative_to(destination)):hashlib.sha256(f.read_bytes()).hexdigest()
             for f in destination.rglob('*') if f.is_file()}
    cpuinfo=Path('/proc/cpuinfo')
    cpu=next((line.split(':',1)[1].strip() for line in cpuinfo.read_text().splitlines()
              if line.startswith('model name')),None) if cpuinfo.exists() else platform.processor()
    (destination/'integrity.json').write_text(json.dumps(dict(files=receipt,platform=platform.platform(),
                                                              cpu_model=cpu),indent=2)+'\n')
    report.write_text(render(analysis,manifest,restore))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('destination',type=Path)
    parser.add_argument('report',type=Path)
    args=parser.parse_args()
    export(args.source,args.destination,args.report)
