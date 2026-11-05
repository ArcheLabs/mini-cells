"""Pinned historical artifacts, independently reconstructed A2, and strict input hashes."""
import json
import subprocess
import torch
from torch import nn
from bootstrap import BASE,ROOT,shared,old_architecture
from .architecture import initial_backbone


def load():return json.loads((BASE/'protocol.json').read_text())


def inspect():
    p=load();history=BASE/'reports/history';record=json.loads((history/'A2-decision.json').read_text())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(91001);reference=shared.phase_c.build_phase_c(shared.phase_c.X4_NAME)
        for stage in reference.stages:stage.iterations=1;stage.step=nn.Parameter(stage.step[:1].detach().clone())
    template=initial_backbone();a=shared.model_hash(template);b=shared.model_hash(reference)
    if a!=b:raise ValueError('independent Native A2 initialization reconstruction differs')
    source_protocol=BASE.parent/'epclm-x4c-001/protocol.json'
    if shared.digest(source_protocol)!=p['baseline_protocol_sha256']:raise ValueError('historical source protocol bytes differ')
    if json.loads((history/'protocol.json').read_text())!=json.loads(source_protocol.read_text()):raise ValueError('published historical protocol content differs')
    pinned=json.loads((history/'provenance.json').read_text())['x4c_source_sha256']
    source_match={name:shared.digest(BASE.parent/'epclm-x4c-001'/name)==value for name,value in pinned.items()}
    if not all(source_match.values()):raise ValueError('historical X4C dependency source changed')
    return {'baseline_run_id':record['run_id'],'baseline_source':p['baseline_source_commit'],
        'historical_backbone_configuration':record['configuration'],'historical_final_weight_identity':record['model_weight_identity'],
        'historical_checkpoint':record['weights'],'historical_initial_weights':'NOT_ARCHIVED_IN_PUBLISHED_ARTIFACTS',
        'initialization_reconstruction_sha256':a,'independent_native_reconstruction_sha256':b,
        'initialization_status':'SOURCE_AND_SEED_RECONSTRUCTION_HASH_VERIFIED; NO_ARCHIVED_INITIAL_HASH',
        'strict_historical_initialization_match':False,'local_torch_version':str(torch.__version__),
        'source_matches':source_match,'source_protocol_sha256':shared.digest(source_protocol),
        'published_protocol_sha256':shared.digest(history/'protocol.json'),'protocol_note':'publication JSON was sorted; source bytes hash and parsed content both verified','historical_evaluation':'archived measured values; not reevaluated without final weights',
        'data_boundary':'Native continuous token stream sampled as independent 125-token examples; no State across samples or long-memory claim'}


def verify_inputs(corpus):
    expected=load()['data'];actual={'dataset_manifest_sha256':shared.sha(corpus.manifest),
        'tokenizer_sha256':shared.digest(corpus.tokenizer_path),'train_sha256':shared.digest(corpus.train_path),
        'validation_sha256':shared.digest(corpus.validation_path)}
    mismatch={k:{'expected':expected[k],'actual':v} for k,v in actual.items() if expected[k]!=v}
    if mismatch:raise ValueError('historical dataset identity mismatch: '+json.dumps(mismatch))
    return actual
