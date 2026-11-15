# %% [markdown]
# # Portable Cell v0.1 runtime
# Enable GPU and Internet. Attach an HF_TOKEN Kaggle secret with access to
# Meta's gated Llama 3.2 1B Instruct model. Run the script once.

# %%
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from kaggle_secrets import UserSecretsClient

remote = 'https://github.com/ArcheLabs/mini-cells.git'
upstream_url = 'https://github.com/sarahpannn/titans-pytorch.git'
upstream_commit = 'c60b4efaa3a3c6df4deebd57cbccf75947b87cad'
os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')
output = Path(tempfile.mkdtemp(prefix='portable-cell-', dir='/kaggle/working')) / 'results'


def run(*args):
    subprocess.run(args, check=True)


with tempfile.TemporaryDirectory(prefix='portable-source-') as directory:
    root = Path(directory)
    repo, upstream = root / 'mini-cells', root / 'titans'
    run('git', 'clone', '--depth', '1', '--branch', 'research/portable-cell-v01', remote, str(repo))
    source = repo / 'research/portable-cell-v01'
    run('git', 'init', str(upstream))
    run('git', '-C', str(upstream), 'fetch', '--depth', '1', upstream_url, upstream_commit)
    run('git', '-C', str(upstream), 'checkout', '--detach', 'FETCH_HEAD')
    run('git', '-C', str(upstream), 'apply', str(source / 'upstream-position.patch'))
    run(sys.executable, '-m', 'pip', 'install', '-q', str(upstream) + '[research]')
    from huggingface_hub import model_info

    model = 'meta-llama/Llama-3.2-1B-Instruct'
    revision = model_info(model, token=os.environ['HF_TOKEN']).sha
    run('git', '-C', str(repo), 'log', '-1', '--format=MiniCells source %H')
    print(f'Results directory: {output}', flush=True)
    run(sys.executable, str(source / 'run_runtime.py'), '--upstream', str(upstream),
        '--output', str(output), '--revision', revision)
