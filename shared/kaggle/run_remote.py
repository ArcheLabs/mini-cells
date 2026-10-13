"""Actions-side exact-version polling and download using the current Kaggle CLI."""
import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urlsplit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prepared = Path(args.prepared)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi();api.authenticate()
    result = api.kernels_push(str(prepared), timeout=3600)
    if result is None or result.error:
        raise RuntimeError('Kaggle push failed: ' + str(result.error if result else 'empty response'))
    parsed = urlsplit(result.url or '')
    if parsed.scheme != 'https' or parsed.hostname not in {'kaggle.com', 'www.kaggle.com'}:
        raise ValueError('Invalid returned Kaggle URL')
    match = re.fullmatch(r'/(?:code/)?(bingjiangli/minicells)/?', parsed.path)
    if not match or not isinstance(result.versionNumber, int) or isinstance(result.versionNumber, bool) or result.versionNumber < 1:
        raise ValueError('Push did not return the original kernel and a valid version')
    reference = match.group(1) + '/' + str(result.versionNumber)
    context = {'url': result.url, 'kernel_ref': reference, 'version': result.versionNumber}
    (prepared / 'push.json').write_text(json.dumps(context, indent=2) + '\n')
    print('Kaggle pushed version ' + str(result.versionNumber) + ': ' + result.url, flush=True)
    terminal = None
    deadline = time.monotonic() + 80 * 60
    while time.monotonic() < deadline:
        response = api.kernels_status(reference)
        status = str(response.status).lower().split('.')[-1]
        print(json.dumps({'kernel_ref': reference, 'status': status}), flush=True)
        if status in {'complete', 'completed', 'error', 'errored', 'failed', 'cancelled', 'canceled'}:
            terminal = status
            break
        time.sleep(30)
    if terminal is None:
        raise TimeoutError('Kaggle queue/execution exceeded Actions wait budget')
    # Preserve failed-run evidence too. The collector determines completeness.
    api.kernels_output(reference, args.output, force=False, quiet=False)
    if terminal not in {'complete', 'completed'}:
        raise RuntimeError('Kaggle execution ended with ' + terminal + '; output preserved')


if __name__ == '__main__':
    main()
