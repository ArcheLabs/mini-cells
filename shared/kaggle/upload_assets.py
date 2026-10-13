"""Upload already prepared CPU inputs; never change an existing snapshot."""
import json
import sys
import time
from pathlib import Path
from requests.exceptions import HTTPError


def main():
    directory = Path(sys.argv[1])
    reference = json.loads((directory / 'input-reference.json').read_text())
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi();api.authenticate()
    try:
        api.dataset_status(reference['dataset_id'])
    except HTTPError as error:
        if error.response.status_code != 404:
            raise
        result = api.dataset_create_new(str(directory), public=False, quiet=False, convert_to_csv=False)
        if str(getattr(result, 'status', '')).lower() == 'error':
            raise RuntimeError('Dataset creation rejected: ' + str(getattr(result, 'error', result)))
    stop = time.monotonic() + 600
    while time.monotonic() < stop:
        status = api.dataset_status(reference['dataset_id'])
        if status == 'ready':
            print(json.dumps(reference));return
        if status in {'error', 'failed'}:
            raise RuntimeError('Dataset creation failed: ' + status)
        time.sleep(10)
    raise TimeoutError('Dataset preparation did not become ready')


if __name__ == '__main__':
    main()
