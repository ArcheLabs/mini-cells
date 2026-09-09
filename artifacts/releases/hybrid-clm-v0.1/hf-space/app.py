import json
from pathlib import Path

import gradio as gr

DATA = json.loads((Path(__file__).parent / "space-data.json").read_text())
METRICS = DATA["metrics"]

def report(alpha: float) -> str:
    measured = "measured" if alpha in (0.0, 1.0) else "visual interpolation only"
    return (
        f"Alpha: {alpha:.3f} ({measured})\n"
        f"Ranking OFF: {METRICS['ranking_off']:.4f}\n"
        f"Ranking ON: {METRICS['ranking_on']:.4f}\n"
        "Zero-state: pass\nRestoration: pass\n"
        "Locality at alpha=1: unresolved under frozen gate"
    )

with gr.Blocks(title="MiniCells HybridCLM") as demo:
    gr.Markdown("# MiniCells HybridCLM\nEngineering Evidence · Formal Validation Pending")
    alpha = gr.Slider(0.0, 1.0, value=1.0, step=0.125, label="Alpha (measured endpoints; intermediate values are interpolation)")
    output = gr.Textbox(report(1.0), label="Frozen report")
    alpha.change(report, inputs=alpha, outputs=output)

demo.launch()
