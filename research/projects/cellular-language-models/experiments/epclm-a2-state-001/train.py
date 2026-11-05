"""Single Kaggle entry; new package stays separate from frozen X4C imports."""
import argparse
from pathlib import Path
import bootstrap
from a2state.runner import run,publish

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('/kaggle/working/epclm-a2-state-001'))
    parser.add_argument('--cache',type=Path,default=Path('/kaggle/working/x4c-native-corpus'))
    parser.add_argument('--execution-seconds',type=float,default=5400)
    parser.add_argument('--publish-only',type=Path,help='existing run folder; publish immutable evidence without training')
    args=parser.parse_args()
    if args.publish_only:
        import json
        decision=json.loads((args.publish_only/'decision.json').read_text())
        if decision.get('format')!='minicells.epclm-a2-state-001.v1':raise ValueError('wrong saved experiment identity')
        print({'publication_status':'REMOTE_SHA_VERIFIED','remote_sha':publish(args.publish_only,'kaggle/'+decision['run_id'])})
    else:run(args)
