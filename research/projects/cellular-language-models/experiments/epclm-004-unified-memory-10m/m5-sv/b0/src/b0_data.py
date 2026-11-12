"""Independent eight-query B0 banks; engineering preparation does not generate Test."""
from pathlib import Path
import json
from b0_models import EXPERIMENT
from data import MQARConfig, episode, digest, file_hash, SPECIAL, TOKENS, dataset_manifest
from e0 import write


def validate_row(row, config):
    positions=row['query_positions'];ids=row['input_ids'];labels=row['labels']
    if len(ids)!=config['length'] or len(positions)!=config['queries']:raise ValueError('episode shape differs')
    bindings={8+k:264+v for k,v in row['bindings']}
    if len({ids[p-1] for p in positions})!=config['queries']:raise ValueError('query keys repeat')
    for i,(label,is_query) in enumerate(zip(labels,row['query_mask'])):
        if is_query:
            if i not in positions or ids[i]!=6 or ids[i-2]!=5 or label!=bindings[ids[i-1]]:raise ValueError('misaligned Query target')
        elif label!=-100:raise ValueError('non-Query supervision')
    first=positions[0]-2
    for p in positions:
        expected=[5,ids[p-1],6]+([7] if p!=positions[-1] else [])
        if ids[p-2:p+2 if p!=positions[-1] else p+1]!=expected:raise ValueError('query tail contains extra input or gold answer')
    if len(ids[first:])!=4*config['queries']-1:raise ValueError('unexpected query tail')


def rows(config, split, seed, count):
    cfg=MQARConfig(**config,seed=seed)
    result=[episode(cfg,split,i) for i in range(count)]
    for row in result:validate_row(row,config)
    return result


def tokenizer_check(expected):
    path=EXPERIMENT/'assets/tokenizer.json';tokenizer=json.loads(path.read_text())
    vocabulary=tokenizer['model']['vocab']
    if file_hash(path)!=expected or len(vocabulary)!=8192:raise ValueError('frozen tokenizer differs')
    if any(vocabulary[token]!=i for i,token in enumerate(TOKENS)):raise ValueError('reserved token IDs differ')
    return {'sha256':expected,'vocab_size':8192,'special_ids':dict(zip(SPECIAL,range(8)))}


def prepare(folder, protocol):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    cfg=protocol['data'];seeds=protocol['seeds'];sizes=protocol['sizes']
    banks={'overfit':rows(cfg,'train',seeds['overfit'],sizes['overfit']),
           'pilot_train':rows(cfg,'train',seeds['pilot_train'],sizes['pilot_train']),
           'validation':rows(cfg,'validation',seeds['validation'],sizes['validation'])}
    if len({tuple(p) for row in banks['overfit'] for p in row['bindings']})!=128:raise ValueError('fixed16 episodes do not have128 unique bindings')
    manifest=dataset_manifest(cfg,{'train':banks['overfit']+banks['pilot_train'],'validation':banks['validation']},protocol['tokenizer_sha256'])
    manifest.update(format='minicells.m5-sv-b0.datasets.v1',tokenizer=tokenizer_check(protocol['tokenizer_sha256']),
        banks={name:{'episodes':len(bank),'input_tokens':sum(len(r['input_ids']) for r in bank),
             'supervised_targets':sum(sum(y!=-100 for y in r['labels']) for r in bank),
             'unique_binding_pairs':len({tuple(p) for r in bank for p in r['bindings']}),
             'content_sha256':digest(bank)} for name,bank in banks.items()},
        test={'status':'RESERVED_NOT_GENERATED_NOT_ACCESSED','episodes':sizes['reserved_test'],
              'data_seed':seeds['reserved_test'],'split_rule':'(key_index+value_index) mod3 ==2'})
    path=folder/'dataset-manifest.json'
    for name,bank in banks.items():
        file=folder/(name+'.json')
        if file.exists() and digest(json.loads(file.read_text()))!=digest(bank):raise ValueError('existing bank content differs')
        if not file.exists():write(file,bank)
    manifest['files']={name+'.json':file_hash(folder/(name+'.json')) for name in banks}
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('B0 dataset identity changed')
    write(path,manifest);return banks,manifest


def seal_test(folder, protocol, *, explicitly_authorized=False):
    """Separate future preparation API; never invoked by B0 review or training."""
    if not explicitly_authorized:raise PermissionError('Test generation is not authorized in this delivery')
    bank=rows(protocol['data'],'test',protocol['seeds']['reserved_test'],protocol['sizes']['reserved_test'])
    folder=Path(folder);path=folder/'sealed-test.json'
    if path.exists() and digest(json.loads(path.read_text()))!=digest(bank):raise ValueError('sealed Test identity differs')
    if not path.exists():write(path,bank)
    write(folder/'sealed-test-manifest.json',{'status':'SEALED_NOT_ACCESSED','content_sha256':digest(bank),
        'file_sha256':file_hash(path),'episodes':len(bank),'evaluation_authorized':False})
