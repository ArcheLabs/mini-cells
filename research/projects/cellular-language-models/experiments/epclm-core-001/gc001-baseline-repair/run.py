"""Dedicated repair launcher; legacy GC-001 entrypoints remain frozen."""
import argparse
from pathlib import Path
import sys
BASE=Path(__file__).resolve().parent
CORE=BASE.parent
ROOT=next(p for p in BASE.parents if (p/'pyproject.toml').exists())
sys.path[:0]=[str(BASE),str(CORE/'implementation'),str(CORE.parent/'native-clm-v0/implementation'),str(ROOT),str(ROOT/'shared/python')]
from repair_pipeline import launch

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batch',choices=['a','b','c'],default='a')
    parser.add_argument('--output',type=Path);parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    launch(ROOT,BASE,args.output or Path('/kaggle/working/epclm-gc001-repair-'+args.batch),args.device,args.batch)
