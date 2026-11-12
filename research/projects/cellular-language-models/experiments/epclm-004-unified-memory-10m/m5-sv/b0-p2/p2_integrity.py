"""Independent complete downloads, member hashes and probability/ledger checks."""
import hashlib
import json
import tarfile
from pathlib import Path
import numpy as np
from p2_dataset import BASE
from b0_integrity import publisher,verify_bundle
from data import file_hash,digest
from e0 import write,load_checkpoint


def download(artifact,destination):
    import requests
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True);package=destination/'bundle.tar.gz'
    if not package.exists() or file_hash(package)!=artifact['sha256']:
        session=requests.Session();session.headers.update({'Authorization':'Bearer '+publisher().load_github_token('GITHUB_TOKEN')})
        url='https://api.github.com/repos/ArcheLabs/mini-cells/releases/assets/'+str(artifact['artifact_id'])
        with session.get(url,headers={'Accept':'application/octet-stream'},stream=True,timeout=(15,300)) as response:
            response.raise_for_status()
            with (destination/'download.pending').open('wb') as f:
                for block in response.iter_content(1024*1024):f.write(block)
        if file_hash(destination/'download.pending')!=artifact['sha256']:raise ValueError('INDEPENDENT_DOWNLOAD_HASH_MISMATCH')
        (destination/'download.pending').replace(package)
    if package.stat().st_size!=artifact['bytes']:raise ValueError('INDEPENDENT_DOWNLOAD_SIZE_MISMATCH')
    with tarfile.open(package) as tar:
        files=json.load(tar.extractfile('sha256-manifest.json'))
        verify_bundle(package,files)
        extracted=destination/'extracted';extracted.mkdir(exist_ok=True)
        for name in [*files,'sha256-manifest.json']:
            relative=Path(name)
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe archive path')
            path=extracted/relative;path.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(name) as source,path.open('wb') as out:
                import shutil
                shutil.copyfileobj(source,out)
    proof={'status':'PASS','sha256':file_hash(package),'bytes':package.stat().st_size,'files_verified':len(files),
        'scope':'All archived files, including checkpoint/probabilities/metrics/report; no trusted status substitution'}
    write(destination/'download-verification.json',proof);return extracted,proof


def verify_payload(folder,protocol,train,val,identity):
    folder=Path(folder);saved=load_checkpoint(folder/'checkpoints/final.pt');steps=protocol['training']['updates']
    if saved['identity']!=identity:raise ValueError('Payload identity differs')
    if not all(isinstance(v,__import__('torch').Tensor) for v in saved['model'].values()):raise ValueError('Non-tensor model state')
    if digest(json.loads((folder/'datasets/training.json').read_text()))!=digest(train) or digest(json.loads((folder/'datasets/validation.json').read_text()))!=digest(val):
        raise ValueError('Archived dataset identity differs')
    ledger=saved['ledger']
    for key,value in {'committed_updates':steps,'input_tokens':steps*512,'processed_tensor_tokens':steps*512,
                      'padding_tokens':0,'supervised_target_tokens':steps*8,'query_target_tokens':steps*8}.items():
        if ledger[key]!=value:raise ValueError('Payload ledger differs')
    if [r['update'] for r in saved['history']]!=list(range(1,steps+1)):raise ValueError('History gaps')
    if any(not np.isfinite(r['loss']) or not np.isfinite(r['gradient_norm']) for r in saved['history']):raise ValueError('Nonfinite history')
    for step in range(1,steps+1):
        if not np.isfinite(saved['optimizer']['b0_parameter_changes'][step]) or saved['optimizer']['b0_parameter_changes'][step]<=0:
            raise ValueError('No real finite optimizer change')
    for npz in (folder/'training/evaluation').glob('*/*/probabilities.npz'):
        with np.load(npz) as a:
            p=a['probabilities'];logp=a['log_probabilities'];gold=a['targets']
            rows=train[:protocol['p2']['training_diagnostic_episodes']] if npz.parent.name=='training' else val
            expected=np.array([y for row in rows for y in row['labels'] if y!=-100])
            if not np.array_equal(gold,expected) or not np.isfinite(p).all() or not np.isfinite(logp).all():raise ValueError('Probability/label mismatch')
            if not np.allclose(p.sum(-1),1,atol=2e-6) or (p<0).any():raise ValueError('Invalid probabilities')
            m=json.loads((npz.parent/'metrics.json').read_text());correct=int((p.argmax(-1)==gold).sum())
            nll=float(-logp[np.arange(len(gold)),gold].mean(dtype=np.float64))
            if correct!=m['correct'] or abs(nll-m['nll'])>2e-5:raise ValueError('Probability metrics disagree')
    return {'status':'PASS','ledger':ledger,'history_steps_verified':steps,'test_accessed':False}


def freeze_bundle(folder):
    import gzip
    folder=Path(folder);files={str(p.relative_to(folder)):file_hash(p) for p in folder.rglob('*')
        if p.is_file() and p.name not in ('sha256-manifest.json','publication.json','bundle.tar.gz') and not p.name.endswith('.pending')}
    path=folder/'sha256-manifest.json'
    if path.exists() and json.loads(path.read_text())!=files:raise ValueError('Immutable P2 run changed')
    write(path,files);target=folder/'bundle.tar.gz'
    if not target.exists():
        aliases={}
        with (folder/'bundle.pending').open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0) as zipped,tarfile.open(fileobj=zipped,mode='w') as tar:
            for name in sorted([*files,'sha256-manifest.json']):
                info=tarfile.TarInfo(name);info.mode=0o644;info.size=(folder/name).stat().st_size
                sha=files.get(name)
                if info.size>1024*1024 and sha in aliases:
                    info.type=tarfile.LNKTYPE;info.linkname=aliases[sha];info.size=0;tar.addfile(info)
                else:
                    if info.size>1024*1024:aliases[sha]=name
                    with (folder/name).open('rb') as stream:tar.addfile(info,stream)
        (folder/'bundle.pending').replace(target)
    verify_bundle(target,files);return target,files

def publish(folder):
    """Reuse the existing publisher with a private, hard-link-aware archive writer."""
    import importlib.util
    import sys
    spec=importlib.util.spec_from_file_location('_p2_private_integrity',BASE.parent/'b0/src/b0_integrity.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    module.freeze_bundle=freeze_bundle
    import requests
    manifest=json.loads((Path(folder)/'run-manifest.json').read_text())
    api='https://api.github.com/repos/ArcheLabs/mini-cells/releases'
    headers={'Authorization':'Bearer '+publisher().load_github_token('GITHUB_TOKEN')}
    response=requests.get(api+'/tags/'+manifest['run_id'],headers=headers,timeout=(15,60))
    if response.status_code==404:
        response=requests.post(api,headers=headers,json={'tag_name':manifest['run_id'],
            'target_commitish':manifest['source_commit'],'name':'B0-P2 '+manifest['arm'],'prerelease':True,
            'body':'Controlled single-seed Learnability Pilot. Full bundle with per-file hashes. Independent download/restore evidence is published in the companion result index. No Test access or H5 confirmation.'},timeout=(15,60))
    response.raise_for_status()
    return module.publish_bundle(folder)
