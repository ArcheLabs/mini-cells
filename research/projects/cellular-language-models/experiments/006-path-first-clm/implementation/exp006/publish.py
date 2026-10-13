"""Immutable 006 archives using the existing tested Git/Askpass branch machinery."""
from __future__ import annotations
import argparse,json,shutil,sys
from pathlib import Path
from .config import BASE,ROOT,load,authorize,digest
from .reuse import canonical_sha256
from exp005 import publish as historical_publisher
sys.path.insert(0,str(ROOT/'scripts/research'))
from publish_experiment_results import run_git
from publish_core_validation_007 import push_branch

def branch_name(source):return 'results/006-path-first-clm-'+source[:12]
def prepare_branch(source,secret):
    previous=historical_publisher.BRANCH_PREFIX
    try:
        historical_publisher.BRANCH_PREFIX='results/006-path-first-clm-'
        branch=historical_publisher.prepare_result_branch(source,secret)
    finally:historical_publisher.BRANCH_PREFIX=previous
    protected=[str(BASE.relative_to(ROOT)/p) for p in ('implementation','config','protocol.json','registration','PROTOCOL_DRAFT.md')]
    if run_git(ROOT,'diff','--quiet',source,'HEAD','--',*protected,check=False).returncode:raise RuntimeError('006 source or frozen contract differs')
    return branch

def archive(output,c,p,seed,role,expected_source=None):
    authorize(c,p,seed,role);output=Path(output)
    record='metrics.json' if (output/'metrics.json').exists() else 'failure.json'
    m=json.loads((output/record).read_text());i=m['identity']
    if expected_source is not None and i['source_commit']!=expected_source:raise ValueError('source does not match result branch')
    if i['experiment_id']!='006-path-first-clm' or i['seed']!=seed or i['role']!=role or i['config_sha256']!=canonical_sha256(c) or i['protocol_sha256']!=canonical_sha256(p) or i['data_sha256']!=(c['data'].get('confirmation_manifest_sha256') if role=='confirmation' else c['data']['manifest_sha256']):raise ValueError('publication contract mismatch')
    if m['status'] not in ('completed','INCOMPLETE'):raise ValueError('invalid archive status')
    names=['metrics.json','learning_curve.json','decode-profile.json','failure.json']
    complete=m['status']=='completed' and (output/'decode-profile.json').exists()
    if (output/'decode-profile.json').exists():
        d=json.loads((output/'decode-profile.json').read_text())
        if d['identity']!=i or d['checkpoint_sha']!=m['checkpoint_sha'] or not d['parity']['passed']:raise ValueError('decode identity/parity mismatch')
    target=BASE/'artifacts'/i['source_commit']/f'seed-{seed}'/i['arm'];files={n:digest(output/n) for n in names if (output/n).is_file()}
    if not complete:target=target.with_name(i['arm']+'-incomplete')/canonical_sha256(files)
    if target.exists():
        manifest=json.loads((target/'evidence.json').read_text())
        if manifest['files']!=files or any(digest(target/n)!=h for n,h in files.items()):raise FileExistsError('immutable archive differs')
        return target
    target.parent.mkdir(parents=True,exist_ok=True);staging=target.with_name(target.name+'.tmp')
    if staging.exists():raise FileExistsError('preserve interrupted archive staging')
    staging.mkdir()
    for n in files:shutil.copy2(output/n,staging/n)
    (staging/'evidence.json').write_text(json.dumps({'identity':i,'files':files,'status':'completed' if complete else 'INCOMPLETE','completed_checks':['train','decode_parity','isolated_profile'] if complete else ['available files only'],'scientific_decision':False},indent=2)+'\n');staging.replace(target);return target

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config');ap.add_argument('--seed',type=int,required=True);ap.add_argument('--role',default='development');ap.add_argument('--source-commit',required=True);ap.add_argument('--output');ap.add_argument('--secret-name',default='GITHUB_TOKEN');ap.add_argument('--preflight-only',action='store_true');args=ap.parse_args()
    c,p,_=load(args.config);authorize(c,p,args.seed,args.role)
    branch=prepare_branch(args.source_commit,args.secret_name)
    if args.preflight_only:return
    if not args.output:ap.error('--output required')
    target=archive(args.output,c,p,args.seed,args.role,expected_source=args.source_commit)
    run_git(ROOT,'config','user.name','MiniCells Kaggle');run_git(ROOT,'config','user.email','research@minicells.invalid');run_git(ROOT,'add','--',str(target.relative_to(ROOT)))
    if run_git(ROOT,'diff','--cached','--quiet',check=False).returncode:run_git(ROOT,'commit','-m','research: archive Path-First 006 arm evidence')
    push_branch(branch,args.secret_name)
if __name__=='__main__':main()
