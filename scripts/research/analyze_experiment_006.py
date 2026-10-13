"""Read-only 006 analysis; no confirmation asset reads."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'research/08-hybrid-clm/experiments/006-path-first-clm/implementation')]
from exp006.config import load,BASE
from exp006.report import analyze,matrix
from exp006.reuse import write_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--source-commit',required=True);ap.add_argument('--config');ap.add_argument('--output',required=True);args=ap.parse_args();c,p,_=load(args.config)
    seeds=[args.seed] if args.seed is not None else c['seeds']['confirmation'];reports=[]
    for seed in seeds:
        rows={}
        for arm in p['arms']:
            folder=BASE/'artifacts'/args.source_commit/f'seed-{seed}'/arm
            if (folder/'metrics.json').exists():
                rows[arm]=json.loads((folder/'metrics.json').read_text())
                if (folder/'decode-profile.json').exists():rows[arm]['decode_profile']=json.loads((folder/'decode-profile.json').read_text())['profile']
        reports.append({'seed':seed,**analyze(rows,c,p)})
    result=reports[0] if args.seed is not None else matrix(reports,c,p);out=Path(args.output)
    if out.exists():raise FileExistsError('analysis record is immutable')
    out.parent.mkdir(parents=True,exist_ok=True);write_json(out,result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
