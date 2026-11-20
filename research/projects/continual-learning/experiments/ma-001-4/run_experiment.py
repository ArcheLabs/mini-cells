"""Deterministic CPU runtime feasibility, no training."""
import argparse
import json
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time

MODES=('inspect','independent-state','multi-write','multi-read','checkpoint','migration','address-integration')
QUERY='Compare GPU-PVM execution, portable cell restart, and Titans Neural Memory recall.'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('upstream',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--mode',choices=MODES+('all','restore-worker'),default='all')
    p.add_argument('--bundle',type=Path)
    p.add_argument('--e5',type=Path)
    p.add_argument('--index',type=Path)
    a=p.parse_args()
    out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,text=True).strip(),
                python=sys.version,platform=platform.platform(),cases={})
    def save():
        report['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (out/'decision.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        start=time.monotonic()
        from runtime import Runtime,torch
        import numpy as np
        torch.set_num_threads(1)
        rt=Runtime(a.upstream)
    except (ImportError,FileNotFoundError,subprocess.CalledProcessError) as exc:
        report.update(status='BLOCKED',reason=str(exc))
        save()
        return 2
    report.update(model=rt.identity,torch=str(torch.__version__),load_seconds=time.monotonic()-start)
    ids=['minijam','minicells','ai_memory']
    rng=np.random.default_rng(42)
    payload=[tuple(rng.normal(size=(3,16)).astype('float32') for _ in range(2)) for _ in range(6)]
    query=rng.normal(size=(1,16)).astype('float32')
    def read(cells):
        before=rt.hashes()
        values=rt.read_many(query,cells)
        assert rt.hashes()==before and all(np.isfinite(v['read_result']).all() for v in values)
        return values
    if a.mode=='restore-worker':
        start=time.monotonic()
        rt.restore(a.bundle)
        report['restore_seconds']=time.monotonic()-start
        initial=rt.hashes()
        cells=ids
        if a.index:
            sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ma-001'/'scripts'))
            from importlib.util import spec_from_file_location, module_from_spec
            spec=spec_from_file_location('ma001_address_runner',Path(sys.path[0])/'run_experiment.py')
            address=module_from_spec(spec)
            spec.loader.exec_module(address)
            Registry,Ranker,embeddings=address.Registry,address.Ranker,address.embeddings
            index=Registry(a.index/'index.sqlite')
            records=index.records()
            index.db.close()
            saved=torch.load(a.index/'index.pt',weights_only=True)
            assert saved['record_ids']==[r['address_id'] for r in records]
            frozen=json.loads((Path(__file__).resolve().parents[1]/'ma-001/artifacts/enriched-cpu/frozen.json').read_text())
            encode=embeddings(False,out,a.e5)
            dense=(encode(['query: '+QUERY])@saved['raw'].T)[0].tolist()
            ranker=Ranker(records,raw=True)
            cells=ranker.route(ranker.scores(QUERY,'E3-Raw',dense),frozen['thresholds']['E3-Raw'])
        first=read(cells)
        rt.write_many({ids[0]:payload[5]})
        report.update(status='PASS',initial=initial,initial_reads=first,continued=rt.hashes(),
                      continued_reads=read(cells),pid=__import__('os').getpid())
        save()
        return 0
    start=time.monotonic()
    for c in ids:
        rt.create(c)
    report['cell_init_seconds']=(time.monotonic()-start)/len(ids)
    rt.write_many(dict(zip(ids,payload)))
    initial=rt.hashes()
    def selected(mode):
        return a.mode in ('all',mode)
    try:
        if selected('inspect'):
            report['cases']['inspect']=dict(status='PASS',state_fields=rt.identity['state_fields'])
        if selected('independent-state'):
            assert len(set(initial.values()))==3
            rt.write_many({ids[0]:payload[3]})
            after=rt.hashes()
            assert after[ids[0]]!=initial[ids[0]] and all(after[c]==initial[c] for c in ids[1:])
            report['cases']['independent-state']=dict(status='PASS',before=initial,after=after)
        if selected('multi-write'):
            before=rt.hashes()
            try:
                rt.write_many({ids[0]:payload[3],ids[1]:payload[4]},fail=ids[1])
            except RuntimeError as exc:
                assert str(exc)=='injected write failure'
            else:
                raise AssertionError('Failure injection did not execute')
            assert rt.hashes()==before
            start=time.monotonic()
            rt.write_many({ids[0]:payload[3],ids[1]:payload[4]})
            after=rt.hashes()
            assert after[ids[2]]==before[ids[2]] and all(before[c]!=after[c] for c in ids[:2])
            report['cases']['multi-write']=dict(status='PASS',seconds=time.monotonic()-start,rollback=True)
        if selected('multi-read'):
            start=time.monotonic()
            values=read(ids[:2])
            assert values[0]['read_result']!=values[1]['read_result']
            report['cases']['multi-read']=dict(status='PASS',seconds=time.monotonic()-start,results=values)
        scenarios=[m for m in ('checkpoint','migration','address-integration') if selected(m)]
        if scenarios:
            bundle=out/'bundle'
            start=time.monotonic()
            rt.save(bundle)
            report['bundle']=dict(seconds=time.monotonic()-start,bytes=sum(f.stat().st_size for f in bundle.iterdir()))
            before=rt.hashes()
            reads=read(ids)
            rt.write_many({ids[0]:payload[5]})
            final,final_reads=rt.hashes(),read(ids)
            for mode in scenarios:
                if mode=='address-integration' and not (a.index and a.e5):
                    report['cases'][mode]=dict(status='BLOCKED',reason='Provide saved MA-001.1 index and pinned E5 snapshot')
                    continue
                target=bundle
                if mode=='migration':
                    target=out/'relocated'/'bundle'
                    shutil.copytree(bundle,target)
                child=out/mode
                cmd=[sys.executable,str(Path(__file__).resolve()),str(a.upstream.resolve()),str(child),
                     '--mode','restore-worker','--bundle',str(target)]
                if mode=='address-integration':
                    cmd+=['--index',str(a.index.resolve()),'--e5',str(a.e5.resolve())]
                start=time.monotonic()
                subprocess.run(cmd,cwd=target.parent,check=True)
                result=json.loads((child/'decision.json').read_text())
                assert result['initial']==before and result['continued']==final
                def keyed(rows):
                    return {r['cell_id']:r for r in rows}
                assert keyed(result['initial_reads'])==keyed(reads)
                assert keyed(result['continued_reads'])==keyed(final_reads)
                report['cases'][mode]=dict(status='PASS',seconds=time.monotonic()-start,
                                            max_error=0,worker=result)
        report['status']='PARTIAL_PASS' if any(r['status']!='PASS' for r in report['cases'].values()) else 'PASS'
    except (AssertionError,RuntimeError,ValueError,subprocess.CalledProcessError) as exc:
        report.update(status='FAIL',reason=repr(exc))
    report['state_hashes']=rt.hashes()
    save()
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    sys.exit(main())
