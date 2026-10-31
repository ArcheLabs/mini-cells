"""Recover exact private saved-output assets using the existing R3b transport."""
import os
import json
from pathlib import Path
import shutil
import sys
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

def recover(root,output,source,protocol):
    if (output/'recovery.json').exists(): return
    sys.path.insert(0,str(root/'research/projects/cellular-language-models/experiments/epclm-002-a0-persistent-state-recall/scripts'))
    from kaggle_output_recovery import _load_backend,_version_number
    api,fetch=_load_backend(root); latest=_version_number(api.metadata())
    cache=Path('/kaggle/temp/core001-recovery')
    incompatible=False
    for version in range(latest,max(0,latest-25),-1):
        token=None; names=set(); seen=set()
        while True:
            page=api.page(version,token); names.update(r['file_name'] for r in page['files'])
            token=page.get('next_page_token')
            if not token: break
            if token in seen: raise ValueError('pagination loop')
            seen.add(token)
        manifests=sorted(n for n in names if n.endswith(output.name+'/recovery.json'))
        for name in manifests:
            inventory={'version_number':version,'files':[{'name':n,'bytes':None} for n in names]}
            destination=cache/str(version); fetch(api,inventory,[name],destination)
            manifest=json.loads((destination/name).read_text()); identity=manifest['identity']
            if identity['source']!=source or identity['protocol']!=protocol:
                incompatible=True
                continue
            prefix=name[:-len('recovery.json')]
            locations=manifest.get('objects', {n:n for n in manifest['files']})
            for relative,physical in locations.items():
                for name_part in (relative,physical):
                    path=Path(name_part)
                    if path.is_absolute() or '..' in path.parts: raise ValueError('unsafe recovery path')
            required=[prefix+locations[n] for n in manifest['files']]
            if not set(required)<=names: raise ValueError('saved bundle incomplete')
            fetch(api,inventory,required,destination)
            for relative,sha in manifest['files'].items():
                path=Path(relative)
                if path.is_absolute() or '..' in path.parts: raise ValueError('unsafe recovery path')
                saved=destination/(prefix+locations[relative])
                if sha256_file(saved)!=sha: raise ValueError('recovery checksum mismatch')
            output.mkdir(parents=True,exist_ok=True)
            for relative in manifest['files']:
                target=output/relative; target.parent.mkdir(parents=True,exist_ok=True)
                source_file=destination/(prefix+locations[relative])
                shutil.copy2(source_file,target)
                if 'objects' in manifest:
                    physical=output/locations[relative]; physical.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(source_file,physical)
            (output/'recovery.json').write_text(json.dumps(manifest))
            return

    if incompatible: raise RuntimeError("Saved Core-001 output belongs to another source/protocol; refusing automatic retraining")
