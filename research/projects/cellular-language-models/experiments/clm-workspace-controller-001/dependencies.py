"""Frozen repository primitives and complete-document assets, not a new data stack."""
import sys,importlib.util
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'epclm-x4c-001'))
import common as shared
sys.path.insert(0,str(BASE)) # Keep this experiment's evaluate/models modules ahead of historical siblings.
ROOT=shared.ROOT

def frozen(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
prior=BASE.parent/'epclm-stateful-computation-routing-001'
primitives=frozen('v02_primitives',prior/'models.py')
assets=frozen('v03_document_assets',prior/'v0.3/data.py')
detach=primitives.detach
