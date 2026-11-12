"""Fail-closed audit of the unchanged, frozen B0 development banks."""
import json
import sys
from collections import Counter
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'src'))
from b0_data import prepare,validate_row
from b0_audit import verify_dependencies
from data import digest
from e0 import write


def audit(protocol,banks,manifest):
    lock=json.loads((BASE/'configs/dataset-lock.json').read_text())
    if manifest['banks']!=lock['banks'] or manifest['files']!=lock['files']:
        raise ValueError('STOP: frozen development data differs')
    if protocol['training']['updates']!=512 or protocol['training']['batch']!=1:
        raise ValueError('STOP: Pilot update/batch budget differs')
    if protocol['data']['length']!=512 or protocol['data']['queries']!=8:
        raise ValueError('STOP: full Episode or Query count differs')
    if protocol['pilot']['validation_updates']!=[128,256,384,512]:
        raise ValueError('STOP: validation schedule differs')
    if protocol['pilot']['input_tokens_per_arm']!=262144 or protocol['pilot']['targets_per_arm']!=4096:
        raise ValueError('STOP: declared token budget differs')
    report={}
    pairs={}
    for name,count,unique in [('pilot_train',512,3746),('validation',64,506)]:
        bank=banks[name]
        if digest(bank)!=lock['banks'][name]['content_sha256']:
            raise ValueError('STOP: bank content differs from frozen identity')
        if len(bank)!=count or len({r['id'] for r in bank})!=count:
            raise ValueError('STOP: episode count/identity differs')
        c=Counter(tuple(pair) for row in bank for pair in row['bindings'])
        if len(c)!=unique:raise ValueError('STOP: distinct binding count differs')
        split='train' if name=='pilot_train' else 'validation'
        for row in bank:
            validate_row(row,protocol['data'])
            if row['split']!=split or sum(y!=-100 for y in row['labels'])!=8:
                raise ValueError('STOP: split/target count differs')
            if any((k+v)%3!=(0 if split=='train' else 1) for k,v in row['bindings']):
                raise ValueError('STOP: binding partition differs')
        pairs[name]=set(c)
        report[name]={'episodes':count,'binding_observations':c.total(),'unique_pairs':len(c),
            'duplicate_observations':c.total()-len(c),'duplicate_fraction':1-len(c)/c.total(),
            'maximum_pair_multiplicity':max(c.values()),'multiplicity_histogram':dict(Counter(c.values())),
            'distinct_keys':len({k for k,v in c}),'distinct_values':len({v for k,v in c}),
            'content_sha256':digest(bank),'input_tokens':count*512,'supervised_targets':count*8}
    overlap=len(pairs['pilot_train']&pairs['validation'])
    if overlap:raise ValueError('STOP: Train/Validation Pair leakage')
    report.update(status='PASS',train_validation_pair_overlap=overlap,
        test_status='RESERVED_NOT_GENERATED_NOT_ACCESSED',
        interpretation='4096 target observations cover3746 distinct training pairs; no independent-example claim',
        generalization='Unseen pairs among shared symbols; not unseen-symbol generalization',
        caveat='Modulo-three partition creates split-dependent conditional value support. Nominal binomial diagnostics are not scientific confirmation.')
    return report


def run(folder,protocol):
    verify_dependencies()
    banks,manifest=prepare(Path(folder)/'datasets',protocol)
    report=audit(protocol,banks,manifest)
    write(Path(folder)/'data-preflight.json',report)
    return banks,manifest,report

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();p=json.loads((BASE/'configs/protocol.json').read_text())
    print(json.dumps(run(args.output,p)[2],indent=2))
