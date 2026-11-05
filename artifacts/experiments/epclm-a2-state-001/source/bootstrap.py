"""Load frozen X4C under its original imports; isolate new modules in a2state.*."""
import importlib
import sys
import types
from pathlib import Path
BASE=Path(__file__).resolve().parent
X4C=BASE.parent/'epclm-x4c-001'
sys.path.insert(0,str(X4C))
shared=importlib.import_module('common')
old_architecture=importlib.import_module('architecture')
old_runtime=importlib.import_module('runtime')
old_train=importlib.import_module('train')
old_evaluate=importlib.import_module('evaluate')
old_flops=importlib.import_module('flops')
if Path(old_architecture.__file__).resolve()!=X4C/'architecture.py':raise RuntimeError('wrong X4C module namespace loaded')
package=types.ModuleType('a2state');package.__path__=[str(BASE)];sys.modules.setdefault('a2state',package)
ROOT=shared.ROOT
