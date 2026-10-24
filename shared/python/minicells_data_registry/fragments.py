"""Reusable exact fragment scan. Hash filtering always ends in elementwise comparison."""
from pathlib import Path
import struct
import subprocess


def compile_scanner(destination):
    source = Path(__file__).with_name('fragment_scan.cpp')
    subprocess.run(['g++', '-O3', '-std=c++17', str(source), '-o', str(destination)], check=True)
    return Path(destination)


def write_patterns(path, sequences):
    with Path(path).open('wb') as stream:
        stream.write(b'MCF1' + struct.pack('<I', len(sequences)))
        for length, blob in sequences:
            if len(blob) != length * 4:
                raise ValueError('PATTERN_LENGTH_MISMATCH')
            stream.write(struct.pack('<I', length) + blob)


def scan(binary, patterns, source, output, budget):
    if Path(output).exists():
        raise FileExistsError('IMMUTABLE_MATCH_OUTPUT_EXISTS')
    subprocess.run([str(binary), str(patterns), str(source), str(output), str(budget)], check=True)


def matches(path):
    with Path(path).open() as stream:
        for line in stream:
            values = [int(x) for x in line.split()]
            yield dict(pattern=values[0], length=values[1], occurrences=values[2],
                       budget_occurrences=values[3], positions=values[4:])
