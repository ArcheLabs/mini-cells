"""Real model acceptance. A fresh process imports and continues the Cell."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from backend import Backbone
from cells import Cells

A = "The Green Belt Movement was founded by Wangari Maathai in Kenya in 1977."
B = "Wangari Maathai received the Nobel Peace Prize in 2004."
QUESTION = "Who founded the Green Belt Movement, and in what year did she win the Nobel Peace Prize?"


def main():
    p = argparse.ArgumentParser()
    for name in ("source", "snapshot", "tokenizer", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    p.add_argument("--restore", action="store_true")
    a = p.parse_args()
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    backbone = Backbone(a.source, a.snapshot, a.tokenizer, dtype=a.dtype)
    cells = Cells(out / ("restored" if a.restore else "original"), backbone)
    if a.restore:
        cell = cells.import_bundle((out / "export.zip").read_bytes())
        assert cell["tensor_sha256"] == json.loads((out/"manifest.json").read_text())["tensor_sha256"]
        selection = [{"cell_id": cell["cell_id"]}]
        answers = {"restored": cells.query("Who founded the Green Belt Movement?", selection, 32, 0)}
        cells.write(cell["cell_id"], B, 512, lambda *_: None, lambda: False)
        answers["updated"] = cells.query(QUESTION, selection, 48, 0)
        (out/"restore.json").write_text(json.dumps(answers, indent=2))
        return
    ids = [cells.create(name)["cell_id"] for name in ("Movement","Prize")]
    for cell, text in zip(ids, (A, B)):
        cells.write(cell, text, 32, lambda *_: None, lambda: False)
    selections = [[{"cell_id": c}] for c in ids]
    qa = {}
    for key, selected in zip(("none","a","b","both"), ([],*selections,sum(selections,[]))):
        before = [cells.read(c)[0]["tensor_sha256"] for c in ids]
        qa[key] = cells.query(QUESTION, selected, 48, 0)
        assert before == [cells.read(c)[0]["tensor_sha256"] for c in ids]
    (out/"queries.json").write_text(json.dumps(qa, indent=2))
    tokens = backbone.tokenizer.encode("Question: Who founded the movement?\nAnswer:", add_special_tokens=False)
    state = cells.read(ids[0])[1]
    original = backbone.step(tokens,[backbone.stream(state)],[1.0])
    from backend import fused_attention
    cache = backbone.stream(state)
    with torch.inference_mode(), fused_attention(backbone.model.model.layers,[cache],[1.0]):
        from_fusion=backbone.model.model(input_ids=torch.tensor([tokens],device="cuda"),
                       past_key_values=cache,use_cache=True,return_dict=True).last_hidden_state
    torch.testing.assert_close(original,from_fusion,atol=0,rtol=0)
    (out/"export.zip").write_bytes(cells.path(ids[0]).read_bytes())
    (out/"manifest.json").write_text(json.dumps(cells.read(ids[0])[0],indent=2))
    (out/"environment.json").write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),
        parameters=backbone.parameters, peak_allocated=torch.cuda.max_memory_allocated(), **backbone.identity),indent=2))
    # Release the parent model before the independent process loads its copy.
    del original, from_fusion, cache, cells, backbone
    torch.cuda.empty_cache()
    subprocess.run([sys.executable,__file__,str(a.source),str(a.snapshot),str(a.tokenizer),
                    str(out),"--dtype",a.dtype,"--restore"],check=True)
    (out/"decision.json").write_text(json.dumps(dict(status="ENGINEERING_CHECKS_COMPLETE",
         knowledge_quality="REVIEW_REQUIRED",single_cell_fusion_exact=True),indent=2))


if __name__=="__main__":
    main()
