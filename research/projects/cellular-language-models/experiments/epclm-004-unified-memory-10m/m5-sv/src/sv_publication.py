"""Load the existing publisher by file, independent of Python's scripts package."""
import importlib.util
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())


def publisher_module():
    name='_minicells_m5_sv_existing_publisher'
    if name not in sys.modules:
        path=ROOT/'scripts/research/publish_experiment_results.py'
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module  # dataclass annotations resolve through the module registry.
        try:spec.loader.exec_module(module)
        except BaseException:sys.modules.pop(name,None);raise
    return sys.modules[name]


if __name__=='__main__':
    module=publisher_module()
    print({'stage':'PUBLICATION_IMPORT_PREFLIGHT','status':'PASS','file':module.__file__},flush=True)
