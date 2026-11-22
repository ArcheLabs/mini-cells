"""Local Web Demo with one serialized GPU worker."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend import Backbone
from cells import Cells


class Create(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class Write(BaseModel):
    text: str = Field(min_length=1)
    chunk_size: int = Field(default=512, ge=1, le=2048)


class Selection(BaseModel):
    cell_id: str
    weight: float = Field(default=1, ge=0, allow_inf_nan=False)


class Query(BaseModel):
    question: str = Field(min_length=1)
    cells: list[Selection] = Field(default_factory=list, max_length=4)
    max_new_tokens: int = Field(default=128, ge=1, le=512)
    temperature: float = Field(default=0, ge=0, le=2)


def app_for(cells):
    app = FastAPI()
    worker = ThreadPoolExecutor(max_workers=1)
    jobs = {}

    async def operation(fn, *args):
        import asyncio
        from functools import partial
        try:
            return await asyncio.get_running_loop().run_in_executor(worker, partial(fn, *args))
        except FileNotFoundError as exc:
            raise HTTPException(404, "Cell not found") from exc
        except (ValueError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/")
    def page():
        return FileResponse(Path(__file__).with_name("index.html"))

    @app.get("/api/model")
    def model_info():
        return dict(parameters=cells.backbone.parameters, **cells.backbone.identity)

    @app.post("/api/cells")
    async def create(body: Create):
        return await operation(cells.create, body.name)

    @app.get("/api/cells")
    async def listing():
        return await operation(lambda: [cells.read(f.stem)[0] for f in sorted(cells.root.glob("*.zip"))])

    @app.post("/api/cells/import")
    async def import_cell(request: Request):
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > 151_000_000:
                raise HTTPException(413, "Cell bundle too large")
        return await operation(cells.import_bundle, bytes(data))

    @app.get("/api/cells/{cell_id}")
    async def details(cell_id: str):
        return (await operation(cells.read, cell_id))[0]

    @app.delete("/api/cells/{cell_id}")
    async def delete(cell_id: str):
        def remove():
            path = cells.path(cell_id)
            path.unlink()
            path.with_suffix(".stream").unlink(missing_ok=True)
        await operation(remove)
        return {"deleted": cell_id}

    @app.post("/api/cells/{cell_id}/fork")
    async def fork(cell_id: str):
        return await operation(cells.fork, cell_id)

    @app.get("/api/cells/{cell_id}/export")
    async def export(cell_id: str):
        from fastapi.responses import Response
        data = await operation(lambda: cells.path(cell_id).read_bytes())
        return Response(data, media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{cell_id}.zip"'})

    @app.post("/api/cells/{cell_id}/write")
    @app.post("/api/cells/{cell_id}/update")
    async def write(cell_id: str, body: Write):
        job_id = str(uuid.uuid4())
        job = {"event": "write_started", "cell_id": cell_id, "cancel": False}
        jobs[job_id] = job

        def run():
            started = time.monotonic()

            def progress(done, total):
                job.update(event="write_progress", processed_tokens=done, total_tokens=total,
                           tokens_per_second=done / max(time.monotonic()-started, 1e-9))
            try:
                result = cells.write(cell_id, body.text, body.chunk_size, progress, lambda: job["cancel"], resume=True)
                job.update(event="write_completed" if result else "write_cancelled", result=result)
            except Exception as exc:
                job.update(event="write_failed", error=str(exc))

        worker.submit(run)
        return {"job_id": job_id}

    @app.delete("/api/progress/{job_id}")
    def cancel(job_id: str):
        if job_id not in jobs:
            raise HTTPException(404, "Write job not found")
        jobs[job_id]["cancel"] = True
        return {"cancel_requested": True}

    @app.get("/api/progress/{job_id}")
    def progress(job_id: str):
        if job_id not in jobs:
            raise HTTPException(404, "Write job not found")
        return jobs[job_id].copy()

    @app.post("/api/query")
    async def query(body: Query):
        return await operation(cells.query, body.question, [c.model_dump() for c in body.cells],
                               body.max_new_tokens, body.temperature)

    return app


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "snapshot", "tokenizer", "storage"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import uvicorn
    model = Backbone(args.source, args.snapshot, args.tokenizer, args.dtype)
    uvicorn.run(app_for(Cells(args.storage, model)), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
