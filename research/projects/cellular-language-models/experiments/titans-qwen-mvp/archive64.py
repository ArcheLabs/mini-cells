"""Recover and verify the frozen Update 64 checkpoint, then archive it."""
import json
from pathlib import Path
import subprocess
import sys

from resume6c import ROOT, recover

if __name__ == '__main__':
    out = Path(sys.argv[1]).resolve()
    checkpoint = recover(out, 'best-adapter.pt')
    baseline = json.loads((ROOT / 'research/titans/stage6d/baseline.json').read_text())
    receipt = json.loads((out / 'recovery.json').read_text())
    if receipt['checkpoint_sha256'] != baseline['checkpoint_sha256']:
        raise ValueError('Checkpoint SHA256 differs from frozen baseline')
    subprocess.run([sys.executable, str(Path(__file__).with_name('verify64.py')),
                    str(checkpoint), str(out)], check=True)
    import importlib.util
    import tarfile

    archive = out / 'stage6c-update64.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        bundle.add(checkpoint, arcname='best-adapter.pt')
        for name in ('recovery.json', 'validation.json'):
            bundle.add(out / name, arcname=name)
    codec = next(ROOT.glob('research/projects/**/epclm-004-unified-memory-10m/m3/archive.py'))
    spec = importlib.util.spec_from_file_location('release_archive', codec)
    uploader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(uploader)
    proof = uploader.upload(archive, dict(
        sha256=uploader.sha256(archive), bytes=archive.stat().st_size,
        checkpoint_sha256=receipt['checkpoint_sha256'], new_training_updates=0,
        validation=json.loads((out / 'validation.json').read_text())), dict(
        archive=dict(repository='ArcheLabs/mini-cells', tag='stage6c-update64-frozen'),
        training_source_commit='f1267f01085c7ad4b75b5e4eec71a4448ac37329'))
    (out / 'archive-receipt.json').write_text(json.dumps(proof, indent=2))
    print(proof, flush=True)
