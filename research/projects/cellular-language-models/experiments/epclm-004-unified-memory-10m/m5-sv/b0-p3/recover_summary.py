"""Read-only arm audit and summary recovery. Never invokes a training worker."""
import argparse
import json
import subprocess
import tarfile
from pathlib import Path
import requests
from p3_data import ROOT, digest, file_hash, write, prepare
from p3_run import verify_payload
from p3_metrics import stage1_gate
from b0_integrity import verify_bundle

BRANCHES = {
    'E0':'kaggle/b0-p3-e0-ecf333523d39205ed4b2',
    'E6':'kaggle/b0-p3-e6-0500486aa3f49400cec9',
    'E7-Innovation':'kaggle/b0-p3-e7-innovation-47b772e8f8bfb538259c',
}


def common_identity(manifests):
    fields=('experiment','source_commit','template_sha256','dataset_sha256','environment_sha256','protocol_sha256','config_sha256')
    if len(manifests)!=3 or any(not isinstance(m,dict) for m in manifests):
        raise TypeError('Exactly three manifest mappings required')
    if {m['arm'] for m in manifests}!=set(BRANCHES):raise ValueError('Arm cohort differs')
    identity={k:manifests[0][k] for k in fields}
    if any(any(m[k]!=v for k,v in identity.items()) for m in manifests):raise ValueError('Arm identities differ')
    return identity


def recover(output):
    output.mkdir(parents=True,exist_ok=True);indexes={};results={};events=[]
    for arm,branch in BRANCHES.items():
        ref='refs/remotes/recovery/'+arm.lower()
        subprocess.run(['git','fetch','git@github.com:ArcheLabs/mini-cells.git',branch+':'+ref],cwd=ROOT,check=True)
        revision=subprocess.check_output(['git','rev-parse',ref],cwd=ROOT,text=True).strip()
        name='artifacts/experiments/epclm004-m5-sv-b0-p3/'+branch.split('/')[-1]+'/decision.json'
        index=json.loads(subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT,text=True));indexes[arm]=index
        write(output/(arm+'-index.json'),{'revision':revision,'index':index})
    identity=common_identity([i['manifest'] for i in indexes.values()])
    for arm,index in indexes.items():
        artifact=index['artifact'];folder=output/arm;folder.mkdir(exist_ok=True);package=folder/'bundle.tar.gz'
        local_archive=Path(__import__('os').environ.get('P3_ARCHIVE_DIRECTORY',str(output)))/(arm+'.tar.gz')
        if not package.exists() and local_archive.exists():
            __import__('shutil').copy2(local_archive,package)
        if not package.exists():
            with requests.get(artifact['download_url'],stream=True,timeout=(20,300)) as response:
                response.raise_for_status()
                with package.with_suffix('.pending').open('wb') as stream:
                    for block in response.iter_content(1024*1024):stream.write(block)
            package.with_suffix('.pending').replace(package)
        if file_hash(package)!=artifact['sha256'] or package.stat().st_size!=artifact['bytes']:raise ValueError('Archive SHA/size mismatch')
        extracted=folder/'extracted';extracted.mkdir(exist_ok=True)
        with tarfile.open(package) as archive:
            hashes=json.load(archive.extractfile('sha256-manifest.json'));verify_bundle(package,hashes)
            for name in [*hashes,'sha256-manifest.json']:
                rel=Path(name)
                if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe archive member')
                target=extracted/rel;target.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(name) as source:target.write_bytes(source.read())
        manifest=json.loads((extracted/'run-manifest.json').read_text());protocol=json.loads((extracted/'protocol.json').read_text())
        if manifest!=index['manifest'] or digest(protocol)!=manifest['protocol_sha256']:raise ValueError('Protocol/manifest mismatch')
        if digest(json.loads((extracted/'config.json').read_text()))!=manifest['config_sha256']:raise ValueError('Configuration hash mismatch')
        stages=[s for s in (1,2) if (extracted/f'stage{s}-result.json').exists()]
        banks, dataset_manifest=prepare(output/'regenerated-data',protocol)
        if dataset_manifest['dataset_sha256']!=manifest['dataset_sha256']:raise ValueError('Regenerated dataset identity differs')
        payload=verify_payload(extracted,banks,protocol,manifest,stages)
        first=json.loads((extracted/'stage1-result.json').read_text());gate=stage1_gate(first['validation'],first['online'],protocol)
        if 2 in stages and index['verification']['stage1_gate']!='PASS':raise ValueError('Unauthorized Stage2')
        results[arm]={'stages':stages,'updates':sum(json.loads((extracted/f'stage{s}-result.json').read_text())['stage_ledger']['committed_updates'] for s in stages),
            'stage1_gate':gate,'payload':payload,'archive_sha256':file_hash(package),'files_verified':len(hashes),
            'original_gpu_restore_proof':index['verification']['restore'],'validation':{str(s):json.loads((extracted/f'stage{s}-result.json').read_text())['validation'] for s in stages}}
        events.extend(index['verification']['charged_worker_phases'])
        print({'arm':arm,'archive':'VERIFIED','stages':stages,'updates':results[arm]['updates']},flush=True)
    summary={'format':'minicells.b0-p3.recovered-summary.v1','identity':identity,'arms':results,'gpu_charged_seconds':sum(e['elapsed_seconds'] for e in events),
        'gpu_phases':events,'new_optimizer_updates':0,'test_accessed':False,'hypotheses':'NOT_EVALUATED',
        'verification_scope':'Fresh archive SHA/member/payload/checkpoint/curve audit; GPU numerical replay proof preserved from original published indexes. No new GPU replay.'}
    write(output/'summary.json',summary)
    report='# B0-P3 recovered summary\n\nNo training was rerun. Test remained sealed. Each archive was freshly SHA256/member-verified and its actual checkpoint, ledger, curves and regenerated Train/Validation data checked. Numerical GPU replay proof is from the original run, not a new recovery replay.\n\n| Arm | Stage1 updates | Stage2 completed | Final candidate EM | Gate |\n|---|---:|---|---:|---|\n'
    for arm,r in results.items():
        report+=f"|{arm}|128|{2 in r['stages']}|{r['validation']['1']['active_exact_match']:.6f}|{r['stage1_gate']['status']}|\n"
    report+='\nCharged GPU phase time: '+str(summary['gpu_charged_seconds'])+' seconds. Stage2 is not authorized when any Stage1 gate blocks. This outcome does not establish scientific efficacy or architectural impossibility.\n\n## Full verified evidence\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n'
    (output/'REPORT.md').write_text(report)
    return summary

def publish_summary(output):
    """Publish a new summary branch only; historical arm refs are never push targets."""
    import tempfile
    import shutil
    summary=json.loads((output/'summary.json').read_text())
    branch='kaggle/b0-p3-summary-recovery-'+digest(summary)[:16]
    remote='git@github.com:ArcheLabs/mini-cells.git'
    with tempfile.TemporaryDirectory(prefix='p3-summary-publish-') as temporary:
        root=Path(temporary)/'repo'
        subprocess.run(['git','clone','--quiet','--shared','--no-checkout',str(ROOT),str(root)],check=True)
        subprocess.run(['git','fetch',remote,'main'],cwd=root,check=True)
        subprocess.run(['git','checkout','--quiet','-b',branch,'FETCH_HEAD'],cwd=root,check=True)
        destination=root/'artifacts/experiments/epclm004-m5-sv-b0-p3'/branch.split('/')[-1]
        destination.mkdir(parents=True)
        for name in ('summary.json','REPORT.md','E0-index.json','E6-index.json','E7-Innovation-index.json'):
            shutil.copy2(output/name,destination/name)
        write(destination/'sha256-manifest.json',{f.name:file_hash(f) for f in destination.iterdir()})
        subprocess.run(['git','add',str(destination.relative_to(root))],cwd=root,check=True)
        subprocess.run(['git','-c','user.name=MiniCells Research','-c','user.email=research@minicells.local','commit','-qm','Recover verified B0-P3 summary without training'],cwd=root,check=True)
        sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
        subprocess.run(['git','push',remote,'HEAD:refs/heads/'+branch],cwd=root,check=True)
        actual=subprocess.check_output(['git','ls-remote',remote,'refs/heads/'+branch],cwd=root,text=True).split()[0]
        if actual!=sha:raise ValueError('Remote summary SHA mismatch')
    proof={'status':'REMOTE_SHA_VERIFIED','branch':branch,'sha':sha,'new_optimizer_updates':0}
    write(output/'summary-publication.json',proof)
    return proof


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--publish',action='store_true')
    args=parser.parse_args();recover(args.output)
    if args.publish:print(publish_summary(args.output),flush=True)
