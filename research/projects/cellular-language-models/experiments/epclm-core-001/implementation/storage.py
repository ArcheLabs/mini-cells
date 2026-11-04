"""Recover exact private saved-output assets using the existing R3b transport."""
import os
import json
from pathlib import Path
import shutil
import sys
import uuid
from native_clm_runtime import sha256_file

def bundle(output,identity):
    # Immutable objects keep the last committed manifest usable if a working
    # checkpoint/report changes immediately before interruption.
    objects = output / 'recovery-objects'
    objects.mkdir(parents=True, exist_ok=True)
    files, locations = {}, {}
    for path in list(output.rglob('*')):
        if not path.is_file() or objects in path.parents or path.name in ('recovery.json','recovery.tmp') or path.suffix == '.tmp':
            continue
        relative = path.relative_to(output).as_posix()
        sha = sha256_file(path)
        physical = objects / sha
        if not physical.exists():
            temporary = objects / (sha + '.tmp')
            shutil.copyfile(path, temporary)
            temporary.replace(physical)
        files[relative] = sha
        locations[relative] = physical.relative_to(output).as_posix()
    temporary = output / 'recovery.tmp'
    with temporary.open('w') as stream:
        json.dump({'identity':identity,'files':files,'objects':locations},stream,indent=2)
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(output/'recovery.json')
    live = set(locations.values())
    for path in objects.iterdir():
        if path.relative_to(output).as_posix() not in live:
            path.unlink()

def asset_identity_matches(manifest,metadata,spec):
    names=('assets/tokenizer.json','assets/train.json','assets/development.json')
    return (metadata.get('dataset_id')==spec['id'] and metadata.get('dataset_revision')==spec['revision']
            and metadata.get('requested_stories')==spec['stories']
            and metadata.get('split')=='core001-content-sha256-mod10-development'
            and manifest['files'].get('assets/manifest.json')==manifest['identity'].get('data')
            and all(metadata['files'].get(n.split('/')[-1])==manifest['files'].get(n) for n in names))


def reuse_local_assets(output,source,protocol,spec):
    record=output/'recovery.json'
    if not record.exists(): return False
    manifest=json.loads(record.read_text()); identity=manifest['identity']
    if identity['source']==source and identity['protocol']==protocol: return True
    if spec is None: raise ValueError('local saved source/protocol differs')
    names=('assets/manifest.json','assets/tokenizer.json','assets/train.json','assets/development.json')
    locations=manifest.get('objects',{n:n for n in manifest['files']})
    for name in names:
        physical=Path(locations[name])
        if physical.is_absolute() or '..' in physical.parts: raise ValueError('unsafe local asset path')
        if sha256_file(output/physical)!=manifest['files'][name]: raise ValueError('local asset checksum mismatch')
    metadata=json.loads((output/locations['assets/manifest.json']).read_text())
    if not asset_identity_matches(manifest,metadata,spec): raise ValueError('local dataset identity differs')
    archive=output.with_name(output.name+'-previous-'+uuid.uuid4().hex[:8])
    output.rename(archive); output.mkdir()
    for name in names:
        target=output/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(archive/locations[name],target)
    proof={'status':'VERIFIED_ASSETS_ONLY','prior_identity':identity,'prior_output_archive':str(archive),
           'asset_sha256':{n:manifest['files'][n] for n in names},'model_optimizer_checkpoint_reused':False}
    (output/'asset-reuse.json').write_text(json.dumps(proof,indent=2)+'\n')
    return True


def recover(root,output,source,protocol,asset_spec=None,cache=None,asset_namespace=None):
    if reuse_local_assets(output,source,protocol,asset_spec): return
    sys.path.insert(0,str(root/'research/projects/cellular-language-models/experiments/epclm-002-a0-persistent-state-recall/scripts'))
    from kaggle_output_recovery import _load_backend,_version_number
    api,fetch=_load_backend(root); latest=_version_number(api.metadata())
    cache=Path(cache) if cache is not None else Path('/kaggle/temp/core001-recovery')
    incompatible=False; reusable=None
    asset_names=('assets/manifest.json','assets/tokenizer.json','assets/train.json','assets/development.json')

    def download(manifest,selected,prefix,version,names,destination):
        locations=manifest.get('objects',{n:n for n in manifest['files']})
        for logical in selected:
            for name in (logical,locations[logical]):
                path=Path(name)
                if path.is_absolute() or '..' in path.parts: raise ValueError('unsafe recovery path')
        required=[prefix+locations[n] for n in selected]
        if not set(required)<=names: raise ValueError('saved bundle incomplete')
        inventory={'version_number':version,'files':[{'name':n,'bytes':None} for n in required]}
        fetch(api,inventory,required,destination)
        files={n:destination/(prefix+locations[n]) for n in selected}
        for name,path in files.items():
            if sha256_file(path)!=manifest['files'][name]: raise ValueError('recovery checksum mismatch')
        return files

    for version in range(latest,max(0,latest-25),-1):
        print('[Core recovery] inspect saved version '+str(version),flush=True)
        token=None; names=set(); seen=set()
        while True:
            page=api.page(version,token); names.update(r['file_name'] for r in page['files'])
            token=page.get('next_page_token')
            if not token: break
            if token in seen: raise ValueError('pagination loop')
            seen.add(token)
        manifests=sorted(n for n in names if (n.endswith(output.name+'/recovery.json') or (asset_namespace is not None and n.endswith(asset_namespace+'/recovery.json'))))
        for name in manifests:
            inventory={'version_number':version,'files':[{'name':name,'bytes':None}]}
            destination=cache/str(version); fetch(api,inventory,[name],destination)
            manifest=json.loads((destination/name).read_text()); identity=manifest['identity']
            prefix=name[:-len('recovery.json')]
            compatible=identity['source']==source and identity['protocol']==protocol and name.endswith(output.name+'/recovery.json')
            if compatible:
                files=download(manifest,manifest['files'],prefix,version,names,destination)
                for relative,saved in files.items():
                    target=output/relative; target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(saved,target)
                    if 'objects' in manifest:
                        physical=output/manifest['objects'][relative]; physical.parent.mkdir(parents=True,exist_ok=True)
                        shutil.copy2(saved,physical)
                (output/'recovery.json').write_text(json.dumps(manifest))
                return
            incompatible=True
            if reusable is not None or asset_spec is None or not set(asset_names)<=manifest['files'].keys():
                continue
            metadata_file=download(manifest,['assets/manifest.json'],prefix,version,names,destination)['assets/manifest.json']
            metadata=json.loads(metadata_file.read_text())
            if not asset_identity_matches(manifest,metadata,asset_spec): continue
            reusable=(download(manifest,asset_names,prefix,version,names,destination),identity,version)

    if reusable is not None:
        files,prior_identity,version=reusable
        for relative,saved in files.items():
            target=output/relative; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(saved,target)
        proof={'status':'VERIFIED_ASSETS_ONLY','prior_identity':prior_identity,'kaggle_version':version,
               'asset_sha256':{n:sha256_file(p) for n,p in files.items()},'model_optimizer_checkpoint_reused':False}
        (output/'asset-reuse.json').write_text(json.dumps(proof,indent=2)+'\n')
        print('[Core recovery] verified tokenizer/data reused; old-source training checkpoints excluded',flush=True)
        return
    if incompatible: raise RuntimeError('Saved Core-001 source/protocol differs and no compatible verified assets were found')
