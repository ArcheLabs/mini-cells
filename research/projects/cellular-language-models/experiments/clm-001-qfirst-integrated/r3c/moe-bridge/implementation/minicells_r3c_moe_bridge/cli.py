import argparse
import json
from .common import configuration, read
from .runner import engineering, gpu_smoke


def main():
    parser = argparse.ArgumentParser(
        description="R3c isolated CPU engineering; GPU is separately authorized"
    )
    parser.add_argument(
        "--mode",
        choices=("engineering", "gpu-smoke", "pilot", "development", "development-worker"),
        default="engineering",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--assets")
    parser.add_argument("--authorization")
    parser.add_argument("--publish-github", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--worker-count", type=int)
    a = parser.parse_args()
    if a.mode == "development":
        from .development import automatic

        print(json.dumps(automatic(a.output, a.assets), indent=2))
        return
    if a.mode == "development-worker":
        from .development import worker

        worker(a.output, a.assets, a.seed, a.worker_index, a.worker_count)
        return
    if a.mode == "pilot":
        from .pilot import automatic

        print(json.dumps(automatic(a.output, a.assets), indent=2))
        return
    if a.mode == "engineering":
        result = engineering(a.output)
    else:
        result = gpu_smoke(
            configuration(), a.output, a.assets, read(a.authorization) if a.authorization else None
        )
    if a.publish_github:
        from .publication import publish, public_summary
        from pathlib import Path

        if a.mode != "engineering":
            raise ValueError("GPU result publication must follow approved release schema")
        report = publish(public_summary(result), Path(a.output) / "metadata-checkout")
        print(json.dumps(report))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("arms", "provenance", "measurements")},
            indent=2,
        )
    )
