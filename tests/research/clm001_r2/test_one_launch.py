import pytest
import torch
from clm001_r2 import cli, data, publication, profiling
from clm001_r2.common import load
from clm001_r2.report import seal


@pytest.mark.parametrize("pilot_pass", [True, False])
def test_one_launch_continues_or_stops_without_manual_phase(monkeypatch, tmp_path, pilot_pass):
    c = load()
    output = tmp_path / "output"
    output.mkdir()
    assets = tmp_path / "assets"
    assets.mkdir()
    phases = []
    monkeypatch.setattr(cli, "cpu_gate", lambda *args: phases.append("cpu"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    def publish(*args, **kwargs):
        phases.append("return-check" if kwargs.get("preflight_only") else "publish")
        return {"publication_status": "PUSHED"}

    monkeypatch.setattr(publication, "publish", publish)
    monkeypatch.setattr(
        data,
        "load_assets",
        lambda *args: {
            "asset_sha256": "a" * 64,
            "tokenizer_sha256": c["data"]["required_r1_tokenizer_sha256"],
        },
    )

    def workers(config, config_path, out, asset_path, mode):
        phases.append(mode)
        if mode == "gpu-pilot":
            for arm in c["arms"]:
                if pilot_pass or arm == "QF-R2":
                    seal(out / "preflight" / arm / "gpu-pilot.json", {"status": "GPU_PILOT_PASS"})
                else:
                    seal(
                        out / "preflight" / arm / "gpu-pilot-failure.json",
                        {"message": "injected functional failure"},
                    )
            return [0, 0 if pilot_pass else 1]
        assert mode == "train-dev"
        return [0, 0]

    monkeypatch.setattr(cli, "workers", workers)
    monkeypatch.setattr(profiling, "completed", lambda *args: phases.append("profile"))

    def aggregate(*args):
        phases.append("analysis")
        return {"status": "COMPLETE" if pilot_pass else "INCOMPLETE"}

    monkeypatch.setattr(cli, "aggregate", aggregate)
    if pilot_pass:
        assert cli.automatic(c, output, assets)["status"] == "COMPLETE"
        assert (
            phases.index("gpu-pilot")
            < phases.index("train-dev")
            < phases.index("profile")
            < phases.index("analysis")
            < phases.index("publish")
        )
        assert phases.count("train-dev") == 1
    else:
        with pytest.raises(RuntimeError, match="GPU Pilot failed"):
            cli.automatic(c, output, assets)
        assert "train-dev" not in phases and "profile" not in phases
        assert phases[-1] == "publish"
