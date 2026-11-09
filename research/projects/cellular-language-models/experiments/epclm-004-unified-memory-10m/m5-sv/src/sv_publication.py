"""Load the existing publisher by file, independent of Python's scripts package."""
import importlib.util
from pathlib import Path
import sys
import subprocess

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


def publication_checkout(source, destination, source_commit):
    """Fetch provenance into this checkout; shared shallow clones can hide fetched objects."""
    if len(source_commit)!=40 or any(c not in '0123456789abcdef' for c in source_commit):
        raise ValueError('invalid publication source commit')
    subprocess.run(['git','clone','--quiet','--shared','--no-checkout',str(source),str(destination)],check=True)
    origin=subprocess.check_output(['git','remote','get-url','origin'],cwd=source,text=True).strip()
    subprocess.run(['git','remote','set-url','origin',origin],cwd=destination,check=True)
    # Verification in the parent checkout is insufficient: shallow boundaries
    # and FETCH_HEAD-only objects are not reliably inherited by a local clone.
    subprocess.run(['git','fetch','--quiet','--depth=1','origin',source_commit],cwd=destination,check=True,timeout=180)
    fetched=subprocess.check_output(['git','rev-parse','FETCH_HEAD'],cwd=destination,text=True).strip()
    if fetched!=source_commit:raise ValueError('publication fetch resolved a different source')
    subprocess.run(['git','checkout','--quiet','--detach',source_commit],cwd=destination,check=True)


if __name__=='__main__':
    module=publisher_module()
    print({'stage':'PUBLICATION_IMPORT_PREFLIGHT','status':'PASS','file':module.__file__},flush=True)
