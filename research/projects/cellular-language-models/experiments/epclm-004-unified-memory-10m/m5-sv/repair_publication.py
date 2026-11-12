"""Recover Version190 metadata and republish it without running an experiment."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys

BASE=Path(__file__).resolve().parent
ROOT=next(p for p in BASE.parents if (p/'pyproject.toml').exists())
CONFIG=json.loads((BASE/'configs/publication-repair-source.json').read_text())
RUN_ID='epclm004-m5-sv-'+CONFIG['source_commit'][:12]


def read(path):return json.loads(Path(path).read_text())

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix('.pending');pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(path)


def selected(relative):
    return relative.suffix in ('.json','.jsonl','.md','.log') and not set(relative.parts)&{'datasets','raw-evaluation'} and relative.name!='publication.json'


def entry_module():
    spec=importlib.util.spec_from_file_location('_m5_sv_publication_repair_entry',BASE/'run.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def verify_original(source):
    decision=read(source/'decision.json');manifest=read(source/'run-manifest.json');identity=decision['identity']
    if decision['format']!='minicells.m5-sv.results.v1' or identity!=manifest['identity']:
        raise ValueError('M5-SV decision/manifest identity differs')
    if identity['source_commit']!=CONFIG['source_commit'] or decision['run_id']!=RUN_ID or decision['branch']!='kaggle/'+RUN_ID:
        raise ValueError('saved Output is not the failed M5-SV run')
    source_sha=identity['source_commit']
    if subprocess.run(['git','cat-file','-e',source_sha+'^{commit}'],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
        subprocess.run(['git','fetch','--depth=1','origin',source_sha],cwd=ROOT,check=True,timeout=120)
    relative=BASE.relative_to(ROOT).as_posix()
    protocol=json.loads(subprocess.check_output(['git','show',source_sha+':'+relative+'/configs/protocol.json'],cwd=ROOT))
    encoded=json.dumps(protocol,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    if hashlib.sha256(encoded).hexdigest()!=identity['config_sha256']:raise ValueError('original protocol hash differs')
    required={'run.py','src/sv_data.py','src/sv_model.py','src/sv_metrics.py','src/sv_analysis.py','tests/test_scientific_protocol.py'}
    if not required<=identity['implementation'].keys():raise ValueError('original implementation manifest is incomplete')
    for name,expected in identity['implementation'].items():
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts:raise ValueError('unsafe implementation path')
        content=subprocess.check_output(['git','show',source_sha+':'+relative+'/'+name],cwd=ROOT)
        if hashlib.sha256(content).hexdigest()!=expected:raise ValueError('original implementation hash differs: '+name)
    return decision


def recover(destination):
    # This is the existing exact-version backend, not a new download protocol.
    archive_path=BASE.parent/'m3/archive.py'
    spec=importlib.util.spec_from_file_location('_m5_sv_reuse_archive_transport',archive_path)
    archive=importlib.util.module_from_spec(spec);spec.loader.exec_module(archive)
    api,inventory,fetch=archive.recovery_backend();inv=inventory(api,CONFIG['version_number'])
    names=[row['name'] for row in inv['files']];cache=destination/'source-cache'
    matches=[n for n in names if n.endswith(RUN_ID+'/decision.json')]
    if len(matches)!=1:raise RuntimeError('VERSION_190_M5_SV_OUTPUT_NOT_UNIQUE: no other version or training attempted')
    prefix=matches[0][:-len('decision.json')]
    fetch(api,inv,[prefix+'decision.json',prefix+'run-manifest.json'],cache)
    source=cache/prefix;verify_original(source)
    needed=[n for n in names if n.startswith(prefix) and selected(Path(n[len(prefix):]))]
    fetch(api,inv,needed,cache)
    proof={'kind':'saved_kaggle_output','kernel':CONFIG['kernel'],'version_number':190,
           'script_version_id':inv.get('metadata',{}).get('script_version_id'),
           'inventory_sha256':inv['inventory_sha256'],'downloaded_metadata_files':len(needed),
           'binary_weights_requested':False,'sealed_test_inputs_requested':False}
    return source,proof


def republish(source, destination, origin):
    decision=verify_original(source);output=destination/'publication'
    output.mkdir(parents=True,exist_ok=True);original_names=[]
    for path in source.rglob('*'):
        relative=path.relative_to(source)
        if path.is_file() and selected(relative):
            target=output/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
            original_names.append(str(relative))
    before={name:sha(output/name) for name in original_names}
    receipt={'format':'minicells.m5-sv.publication-repair.v1','original_run_id':decision['run_id'],
        'original_source_commit':decision['identity']['source_commit'],'original_status':decision['status'],
        'original_error':decision.get('error'),'source':origin,'new_optimizer_updates':0,
        'checkpoint_tensors_loaded':False,'experiment_reexecuted':False,'original_metadata_sha256':before}
    write(output/'publication-repair.json',receipt)
    print({'stage':'PUBLICATION_ONLY_SOURCE_VERIFIED','original_experiment_status':decision['status'],
           'original_error':decision.get('error'),'new_optimizer_updates':0},flush=True)
    publisher=entry_module();proof=publisher.publish(output,decision)
    for name,expected in before.items():
        if sha(output/name)!=expected:raise ValueError('publication retry changed original evidence: '+name)
    receipt.update(publication=proof['status'],remote_commit=proof['commit'],branch=proof['branch'])
    write(destination/'repair-receipt.json',receipt)
    print({'status':'PUBLICATION_REPAIR_COMPLETE','original_experiment_status':decision['status'],
           'original_error':decision.get('error'),'publication':proof['status'],'branch':proof['branch'],
           'remote_commit':proof['commit'],'new_optimizer_updates':0},flush=True)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--destination',type=Path,required=True);args=parser.parse_args()
    args.destination.mkdir(parents=True,exist_ok=True)
    try:
        source,origin=recover(args.destination);republish(source,args.destination,origin)
    except Exception as error:
        write(args.destination/'repair-error.json',{'status':'PUBLICATION_REPAIR_BLOCKED',
            'error':f'{type(error).__name__}: {error}','version_number':190,'new_optimizer_updates':0})
        raise
