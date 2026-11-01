"""Reuse Kaggle's existing authenticated transport and immutable output manifests."""
import json
import signal
import time
from contextlib import contextmanager
from pathlib import Path
import shutil
import subprocess
import sys
from frozen_model import AssetUnavailable,BaselineIntegrityError,check_deadline,safe_path
from native_clm_runtime import sha256_file


def public_result(root,branch,artifact,source=None,protocol=None,commit=None,optional=False):
    ref=subprocess.run(['git','ls-remote','--heads','origin','refs/heads/'+branch],cwd=root,capture_output=True,text=True,timeout=60)
    if ref.returncode:raise AssetUnavailable('GitHub result lookup failed; no completed work will be retrained')
    if not ref.stdout.strip():
        if optional:return None
        raise AssetUnavailable('required published batch missing: '+branch)
    revision=ref.stdout.split()[0]
    if commit is not None and revision!=commit:raise BaselineIntegrityError('historical result branch moved from pinned commit')
    result=subprocess.run(['git','fetch','--depth=1','origin','refs/heads/'+branch],cwd=root,capture_output=True,text=True,timeout=60)
    if result.returncode:raise AssetUnavailable('GitHub result fetch failed')
    fetched=subprocess.check_output(['git','rev-parse','FETCH_HEAD'],cwd=root,text=True).strip()
    if fetched!=revision:raise BaselineIntegrityError('result branch changed during fetch')
    raw=subprocess.check_output(['git','show',fetched+':'+artifact+'/decision.json'],cwd=root)
    value=json.loads(raw)
    if source is not None and value['identity']['source']!=source:raise BaselineIntegrityError('batch source differs')
    if protocol is not None and value['identity']['protocol']!=protocol:raise BaselineIntegrityError('batch protocol differs')
    return value,{'branch':branch,'commit':revision,'decision_sha256':__import__('hashlib').sha256(raw).hexdigest()}


def recover_namespace(root,output,namespace,identity,deadline,max_versions=25,selected=None):
    """Exact-source progress or explicitly pinned historical weights, never an implicit optimizer reuse."""
    sys.path.insert(0,str(root/'research/projects/cellular-language-models/experiments/epclm-002-a0-persistent-state-recall/scripts'))
    from kaggle_output_recovery import _load_backend,_version_number
    api,fetch=_load_backend(root);latest=_version_number(api.metadata())
    cache=Path('/kaggle/temp/gc001-repair-recovery')/namespace
    for version in range(latest,max(0,latest-max_versions),-1):
        check_deadline(deadline);token=None;names=set();seen=set()
        while True:
            check_deadline(deadline);page=api.page(version,token)
            names.update(row['file_name'] for row in page['files']);token=page.get('next_page_token')
            if not token:break
            if token in seen:raise BaselineIntegrityError('pagination loop')
            seen.add(token)
        for name in sorted(n for n in names if n.endswith(namespace+'/recovery.json')):
            safe_path(name);destination=cache/str(version)
            fetch(api,{'version_number':version,'files':[{'name':name,'bytes':None}]},[name],destination)
            manifest=json.loads((destination/name).read_text())
            if manifest['identity']!=identity:continue
            logical=list(manifest['files']) if selected is None else list(selected)
            if not set(logical)<=manifest['files'].keys():raise AssetUnavailable('matching saved bundle lacks required files')
            prefix=name.rsplit('/',1)[0]+'/'
            locations={n:prefix+str(safe_path(manifest.get('objects',{}).get(n,n))) for n in logical}
            for n in logical:
                safe_path(n)
                if Path(n).suffix not in ('.pt','.json','.md','.csv'):raise BaselineIntegrityError('unexpected repair artifact')
                if selected is not None and selected[n]!=manifest['files'][n]:raise BaselineIntegrityError('pinned historical artifact hash differs')
            if not set(locations.values())<=names:raise AssetUnavailable('saved bundle incomplete')
            required=list(dict.fromkeys(locations.values()));check_deadline(deadline)
            fetch(api,{'version_number':version,'files':[{'name':n,'bytes':None} for n in required]},required,destination)
            for n in logical:
                check_deadline(deadline);saved=destination/locations[n]
                if sha256_file(saved)!=manifest['files'][n]:raise BaselineIntegrityError('saved repair artifact checksum differs')
                target=output/safe_path(n);target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(saved,target)
            return {'version':version,'namespace':namespace,'identity':identity,'files':{n:manifest['files'][n] for n in logical}}
    return None


@contextmanager
def timed_budget(seconds):
    """Nested deadlines restore the enclosing timer instead of cancelling it."""
    previous=signal.getitimer(signal.ITIMER_REAL);began=time.monotonic()
    def expired(signum,frame):raise TimeoutError('repair execution/recovery budget exhausted')
    old=signal.signal(signal.SIGALRM,expired)
    limit=min(seconds,previous[0]) if previous[0]>0 else seconds
    signal.setitimer(signal.ITIMER_REAL,max(limit,1e-6))
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
        if previous[0]>0:signal.setitimer(signal.ITIMER_REAL,max(1e-6,previous[0]-(time.monotonic()-began)),previous[1])
