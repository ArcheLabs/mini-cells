"""Engineering preparation only; training flags and budgets are intentionally absent."""
import argparse
import sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
CORE=BASE.parent
ROOT=next(p for p in BASE.parents if (p/'pyproject.toml').exists())
sys.path[:0]=[str(BASE),str(CORE/'gc001-baseline-repair'),str(CORE/'implementation'),str(CORE.parent/'native-clm-v0/implementation'),str(ROOT),str(ROOT/'shared/python')]
from preparation import prepare,write


def publish(output):
    import json,shutil
    from scripts.research.publish_experiment_results import EXPERIMENTS,prepare_artifacts,push_results
    experiment='epclm-gc002-preparation';destination=ROOT/'results'/experiment/'publish';destination.mkdir(parents=True,exist_ok=True)
    decision=json.loads((output/'reports/research-decision.json').read_text())
    write(destination/'decision.json',{'format':'minicells.gc002.preparation.v1','status':decision['result']['status'],**decision})
    mapping={'audit/baseline-audit.json':'baseline-audit.json','audit/data-audit.json':'data-audit.json','audit/tokenizer-audit.json':'tokenizer-audit.json','audit/parameter-match-report.json':'parameter-match-report.json','profiling/kv-cache-equivalence.json':'kv-cache-equivalence.json','profiling/inference-profile.json':'inference-profile.json','profiling/memory-profile.json':'memory-profile.json','profiling/training-flops.json':'training-flops.json','reports/summary.md':'summary.md','audit/token-exposure.json':'token-exposure.json','audit/reused-engineering-evidence.json':'reused-engineering-evidence.json'}
    for source,target in mapping.items():shutil.copyfile(output/source,destination/target)
    shutil.copyfile(BASE/'freeze-candidate.json',destination/'freeze-candidate.json')
    shutil.copyfile(BASE/'RUN-PLAN.zh-CN.md',destination/'run-plan.zh-CN.md')
    artifacts=prepare_artifacts(ROOT,experiment,EXPERIMENTS[experiment],None)
    branch='kaggle/epclm-gc002-preparation-'+decision['identity']['git_commit'][:12]
    push_results(ROOT,artifacts,experiment,branch,'GITHUB_TOKEN','origin/main')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Phase 0/1 only; NOT AUTHORIZED TO TRAIN')
    parser.add_argument('--prepare-only',action='store_true',help='explicit engineering-only intent (also the default)')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--output',type=Path,default=Path('/tmp/gc002-preparation'))
    parser.add_argument('--publish',action='store_true')
    parser.add_argument('--recover-input-assets',action='store_true',help='verify known-version Tokenizer/data only, never weights')
    args=parser.parse_args()
    from repair_io import timed_budget
    with timed_budget(5400):result=prepare(ROOT,BASE,args.output,args.device,recover_assets=args.recover_input_assets)
    if args.publish:
        with timed_budget(600):publish(args.output)
    print(result)
