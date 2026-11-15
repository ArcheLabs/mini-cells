"""Discover the isolated research suite without leaking its module names."""
from pathlib import Path
import subprocess
import sys


def test_persistent_cognitive_suite():
    directory = Path(__file__).resolve().parents[3] / 'research/persistent-cognitive'
    subprocess.run([sys.executable, '-m', 'pytest', '-q', str(directory)],
                   cwd=directory, check=True, timeout=900)
