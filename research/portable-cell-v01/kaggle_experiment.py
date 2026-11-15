"""Prepare the pinned Titan-Llama environment and execute the experiment."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def run(*args):
    subprocess.run(args, check=True)


def digest(path):
    with path.open('rb') as handle:
        h = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
        return h.hexdigest()


def main(output):
    source = Path(__file__).resolve().parent
    upstream_url = 'https://github.com/sarahpannn/titans-pytorch.git'
    upstream_commit = 'c60b4efaa3a3c6df4deebd57cbccf75947b87cad'
    model = 'unsloth/Llama-3.2-1B-Instruct'
    with tempfile.TemporaryDirectory(prefix='portable-upstream-') as directory:
        upstream = Path(directory) / 'titans'
        run('git', 'init', str(upstream))
        run('git', '-C', str(upstream), 'fetch', '--depth', '1', upstream_url, upstream_commit)
        run('git', '-C', str(upstream), 'checkout', '--detach', 'FETCH_HEAD')
        run('git', '-C', str(upstream), 'apply', str(source / 'upstream-position.patch'))
        run(sys.executable, '-m', 'pip', 'install', '-q', str(upstream) + '[research]')
        # This text-only run does not use torchvision. Kaggle's preinstalled
        # extension can target a different Torch ABI after upstream pins Torch.
        for command in (
                [sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchvision'],
                [sys.executable, '-c',
                 'import torch; from transformers import LlamaForCausalLM; '
                 'assert torch.cuda.is_available(), "CUDA unavailable"; '
                 'print(torch.__version__, torch.cuda.get_device_name(0), LlamaForCausalLM.__name__)'],
            ):
            subprocess.run(command, check=True)
        from huggingface_hub import snapshot_download

        revision = '5a8abab4a5d6f164389b1079fb721cfab8d7126c'
        model_path = Path(snapshot_download(model, revision=revision,
                                            allow_patterns=['*.json', '*.safetensors', '*.model', '*.jinja']))
        model_files = {p.name: digest(p) for p in sorted(model_path.iterdir()) if p.is_file()}
        fingerprint = hashlib.sha256(json.dumps(model_files, sort_keys=True).encode()).hexdigest()
        identity = dict(model_id=model, revision=revision, backbone_fingerprint=fingerprint,
                        titan_source_commit=upstream_commit,
                        upstream_config_sha256=digest(upstream / 'configs/seg512_pretrain/training_config.json'),
                        runtime_precision='float32', use_flex_attn=False, use_flash_attn=False)
        candidates = []
        outer = []
        for path in sorted(Path('/kaggle/input').rglob('checkpoint_manifest.json')):
            try:
                record = json.loads(path.read_text())
                weights = path.with_name('outer.safetensors')
                compatible = all(record.get(k) == v for k, v in identity.items())
                compatible = compatible and weights.is_file() and digest(weights) == record.get('outer_sha256')
                if compatible and not outer:
                    outer = ['--outer-checkpoint', str(weights), '--outer-sha256', record['outer_sha256']]
                candidates.append(dict(path=str(path), identity_and_hash_match=compatible))
            except (ValueError, OSError):
                candidates.append(dict(path=str(path), identity_and_hash_match=False))
        (output.parent / 'source_audit.json').write_text(json.dumps(
            dict(identity=identity, model_files=model_files, mounted_candidates=candidates), indent=2))
        run(sys.executable, str(source / 'run_runtime.py'), '--upstream', str(upstream),
            '--output', str(output), '--revision', revision, '--model-path', str(model_path),
            '--backbone-fingerprint', fingerprint, *outer)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
