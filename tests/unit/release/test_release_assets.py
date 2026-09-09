import json
from pathlib import Path

from scripts.release.render_hybrid_clm_release_assets import render


def test_release_assets_use_frozen_evidence(tmp_path: Path) -> None:
    config = json.loads(Path("artifacts/releases/hybrid-clm-v0.1/release-config.json").read_text())
    report = render(config, Path.cwd(), tmp_path)
    assert report["status"] == "ASSETS_RENDERED"
    data = json.loads((tmp_path / "space-data.json").read_text())
    assert data["metrics"]["ranking_off"] == 0.0625
    assert data["metrics"]["ranking_on"] == 0.8203125
    assert data["formal_execution_started"] is False
