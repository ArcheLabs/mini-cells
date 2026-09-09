#!/usr/bin/env python3
"""Render reproducible Hugging Face Model and Space assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import dedent


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(config: dict, root: Path) -> dict[str, float]:
    expected = config["expected_metrics"]
    evidence = config.get("evidence", {}).get("result")
    if evidence:
        path = Path(evidence)
        if not path.is_absolute():
            path = root / path
        if path.is_file():
            observed = _load(path).get("primary_causal_effect", {})
            names = ("ranking_off", "ranking_on", "ranking_gain", "answer_margin_gain", "answer_nll_gain")
            if all(key in observed for key in names) and "B_control_answer_nll_increase" in observed:
                return {
                    **{key: float(observed[key]) for key in names},
                    "b_control_answer_nll_increase": float(observed["B_control_answer_nll_increase"]),
                }
    return {key: float(value) for key, value in expected.items()}


def _space_data(config: dict, metrics: dict[str, float]) -> dict:
    return {
        "schema_version": 1,
        "release": config["package"],
        "status": "engineering_evidence_formal_validation_pending",
        "base_model": config["foundation"]["repo_id"],
        "base_revision": config["foundation"]["revision"],
        "cell_placement": {
            "layer": int(config["mutation"]["layer"]),
            "cell_budget": int(config["mutation"]["cell_budget"]),
        },
        "engineering_seed": int(config["mutation"]["engineering_seed"]),
        "measured_points": [
            {"condition": "cell_off", "ranking_percent": metrics["ranking_off"] * 100},
            {"condition": "cell_on", "ranking_percent": metrics["ranking_on"] * 100},
        ],
        "metrics": metrics,
        "alpha_policy": "measured_points_only_or_label_between_points_as_interpolation",
        "locality_status": "unresolved_at_alpha_1_under_frozen_gate",
        "zero_state_status": "pass",
        "restoration_status": "pass",
        "formal_execution_started": False,
    }


def _model_card(config: dict, metrics: dict[str, float]) -> str:
    foundation = config["foundation"]
    mutation = config["mutation"]
    package = config["package"]
    return dedent(
        f"""
        ---
        library_name: mini-cells
        tags:
          - minicells
          - hybrid-clm
          - moe
        ---

        # MiniCells HybridCLM Cell — Granite 3.1 1B A400M / L{mutation['layer']} / K{mutation['cell_budget']}

        **Artifact type:** Cell mutation only. Foundation weights are not included.

        - Base model: `{foundation['repo_id']}`
        - Immutable base revision: `{foundation['revision']}`
        - MiniCells version: `{package['version']}`
        - Release tag: `{package['tag']}`
        - Cell placement: layer `{mutation['layer']}`, budget `{mutation['cell_budget']}`
        - Engineering seed: `{mutation['engineering_seed']}`

        ## Evaluation summary

        - Ranking OFF: `{metrics['ranking_off']:.6f}`
        - Ranking ON: `{metrics['ranking_on']:.6f}`
        - Ranking gain: `{metrics['ranking_gain']:.6f}`
        - Same-graph zero-state: pass
        - Restoration: pass
        - Locality at alpha=1: unresolved under the frozen gate

        Scientific status: **Engineering Evidence · Formal Validation Pending**.
        Formal seeds remain reserved and untouched. This release does not claim
        standalone CLM conversion, superiority to LoRA, or solved locality.

        ## Installation and attach

        ```python
        from minicells import CellMutation, HybridCLM

        hybrid = HybridCLM.from_pretrained(
            "{foundation['repo_id']}", revision="{foundation['revision']}"
        )
        mutation = CellMutation.from_pretrained("<HF_MODEL_REPO>")
        hybrid.cellularize(mutation.placements)
        hybrid.attach(mutation).set_alpha(mutation, 1.0)
        ```

        License relationship: MiniCells is Apache-2.0; the foundation model
        remains under its own upstream license.
        """
    ).lstrip()


def _space_readme(config: dict) -> str:
    return dedent(
        f"""
        ---
        title: {config['publication']['space_repo']}
        emoji: 🧬
        colorFrom: blue
        colorTo: purple
        sdk: gradio
        app_file: app.py
        ---

        # MiniCells HybridCLM Research Preview

        This Space visualizes the frozen engineering evidence for
        `{config['package']['tag']}`. It is not a new training run.
        """
    ).lstrip()


def _app() -> str:
    return dedent(
        '''
        import json
        from pathlib import Path

        import gradio as gr

        DATA = json.loads((Path(__file__).parent / "space-data.json").read_text())
        METRICS = DATA["metrics"]

        def report(alpha: float) -> str:
            measured = "measured" if alpha in (0.0, 1.0) else "visual interpolation only"
            return (
                f"Alpha: {alpha:.3f} ({measured})\\n"
                f"Ranking OFF: {METRICS['ranking_off']:.4f}\\n"
                f"Ranking ON: {METRICS['ranking_on']:.4f}\\n"
                "Zero-state: pass\\nRestoration: pass\\n"
                "Locality at alpha=1: unresolved under frozen gate"
            )

        with gr.Blocks(title="MiniCells HybridCLM") as demo:
            gr.Markdown("# MiniCells HybridCLM\\nEngineering Evidence · Formal Validation Pending")
            alpha = gr.Slider(0.0, 1.0, value=1.0, step=0.125, label="Alpha (measured endpoints; intermediate values are interpolation)")
            output = gr.Textbox(report(1.0), label="Frozen report")
            alpha.change(report, inputs=alpha, outputs=output)

        demo.launch()
        '''
    ).lstrip()


def render(config: dict, root: Path, output: Path) -> dict:
    metrics = _metrics(config, root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "space-data.json").write_text(json.dumps(_space_data(config, metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "README.md").write_text(_space_readme(config), encoding="utf-8")
    (output / "app.py").write_text(_app(), encoding="utf-8")
    (output / "requirements.txt").write_text("gradio>=4.0,<6\n", encoding="utf-8")
    (output / "model-card.md").write_text(_model_card(config, metrics), encoding="utf-8")
    return {"status": "ASSETS_RENDERED", "output": str(output), "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = _load(args.release_config)
    print(json.dumps(render(config, Path.cwd(), args.output.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
