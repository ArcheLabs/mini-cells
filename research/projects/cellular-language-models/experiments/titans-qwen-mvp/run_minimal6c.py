"""Keep Kaggle transport imports separate from the Qwen runtime."""
from pathlib import Path
import subprocess
import sys

from resume6c import recover

if __name__ == '__main__':
    report = Path(sys.argv[1])
    checkpoint = recover(report.parent, 'best-adapter.pt')
    subprocess.run([sys.executable, str(Path(__file__).with_name('minimal6c.py')),
                    str(report), str(checkpoint)], check=True)
