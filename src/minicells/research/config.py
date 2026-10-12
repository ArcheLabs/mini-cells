"""One declarative experiment definition; historical protocols stay immutable."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def load_definition(path):
    path = Path(path).resolve()
    # JSON is a YAML 1.2 subset, matching the existing repository convention.
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('schema_version') != 2 or not re.fullmatch(r'00[1-4]-[a-z-]+', value['id']):
        raise ValueError('Unknown research definition schema or experiment identity')
    if value.get('runtime') == 'archive':
        raise ValueError('Historical definition is for archive navigation; use its original entrypoint')
    if value['status'] == 'planned':
        raise ValueError('Planned experiments need a registered protocol before execution')
    config = ROOT / value['config']
    protocol = ROOT / value['protocol']
    for item in (config, protocol):
        if not item.is_relative_to(ROOT) or not item.is_file():
            raise ValueError('Definition must reference existing repository files')
    return value, json.loads(config.read_text()), protocol


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
