"""Prepare/upload an immutable CPU asset snapshot from Actions."""
import argparse
import json
import sys
from pathlib import Path
ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / 'pyproject.toml').is_file())
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'shared/python'), str(ROOT)]
from minicells.research.assets import prepare
from minicells.research.config import digest, dump


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--definition', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--owner', default='bingjiangli')
    args = parser.parse_args()
    prepare(args.definition, args.output)
    directory = Path(args.output)
    checksum = digest(directory / 'assets-manifest.json')
    dataset_id = args.owner + '/minicells-assets-' + checksum[:20]
    dump(directory / 'dataset-metadata.json', {'id': dataset_id,
         'title': 'MiniCells assets ' + checksum[:20], 'licenses': [{'name': 'other'}]})
    dump(directory / 'input-reference.json', {'dataset_id': dataset_id,
         'assets_manifest_sha256': checksum, 'immutable_snapshot': True})
    print(json.dumps({'dataset_id': dataset_id, 'assets_manifest_sha256': checksum}))


if __name__ == '__main__':
    main()
