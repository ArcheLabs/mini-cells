"""Local web adapter for MiniCells Python Runtime."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from minicells_native.errors import CellAlreadyExistsError, CellNotFoundError, MiniCellsError
from minicells_native.sdk import DEFAULT_MODEL, MiniCells
from minicells_native.storage import MAX_BUNDLE_BYTES


class Create(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class Write(BaseModel):
    text: str = Field(min_length=1)
    chunk_size: int | str = "auto"


class Selection(BaseModel):
    cell_id: str
    weight: float = Field(default=1, ge=0, allow_inf_nan=False)


class Query(BaseModel):
    question: str = Field(min_length=1)
    cells: str | list[str | Selection] = "auto"
    max_new_tokens: int = Field(default=128, ge=1, le=512)
    temperature: float = Field(default=0, ge=0, le=2)


def cell_json(cell):
    return {**asdict(cell), "cell_id": cell.id}


def app_for(runtime):
    if not isinstance(runtime, MiniCells):
        # Old callers may still hand the adapter a native Cell store.
        runtime = MiniCells._from_backbone(runtime.backbone, runtime.root)
    worker = ThreadPoolExecutor(max_workers=1)
    jobs = {}

    @asynccontextmanager
    async def lifespan(app):
        yield
        worker.shutdown(wait=True)
        runtime.close()

    app = FastAPI(lifespan=lifespan)
    cells = runtime.cells

    async def operation(fn, *args, **kwargs):
        import asyncio
        from functools import partial
        try:
            return await asyncio.get_running_loop().run_in_executor(worker, partial(fn, *args, **kwargs))
        except CellNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except CellAlreadyExistsError as exc:
            raise HTTPException(409, str(exc)) from exc
        except (ValueError, MiniCellsError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/")
    def page():
        return FileResponse(Path(__file__).with_name("index.html"))

    @app.get("/api/model")
    def model_info():
        return runtime.model_info

    @app.post("/api/cells")
    async def create(body: Create):
        return cell_json(await operation(cells.create, body.name))

    @app.get("/api/cells")
    async def listing():
        return [cell_json(item) for item in await operation(cells.list)]

    @app.post("/api/cells/import")
    async def import_cell(request: Request):
        with TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "cell.zip"
            size = 0
            with bundle.open("wb") as out:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_BUNDLE_BYTES:
                        raise HTTPException(413, "Cell bundle too large")
                    out.write(chunk)
            return cell_json(await operation(cells.import_cell, bundle))

    @app.get("/api/cells/{cell_id}")
    async def details(cell_id: str):
        return cell_json(await operation(cells.get, cell_id))

    @app.delete("/api/cells/{cell_id}")
    async def delete(cell_id: str):
        await operation(cells.delete, cell_id)
        return {"deleted": cell_id}

    @app.post("/api/cells/{cell_id}/fork")
    async def fork(cell_id: str):
        return cell_json(await operation(cells.fork, cell_id))

    @app.get("/api/cells/{cell_id}/export")
    async def export(cell_id: str):
        with TemporaryDirectory() as tmp:
            path = await operation(cells.export, cell_id, Path(tmp) / "cell.zip")
            return Response(path.read_bytes(), media_type="application/zip",
                            headers={"Content-Disposition": f'attachment; filename="{cell_id}.zip"'})

    @app.post("/api/cells/{cell_id}/write")
    @app.post("/api/cells/{cell_id}/update")
    async def write(cell_id: str, body: Write):
        await operation(cells.get, cell_id)
        await operation(runtime._chunk_size, body.chunk_size)
        job_id = str(uuid.uuid4())
        job = {"event": "write_started", "cell_id": cell_id, "cancel": False}
        jobs[job_id] = job

        def run():
            started = time.monotonic()

            def progress(done, total):
                job.update(event="write_progress", processed_tokens=done, total_tokens=total,
                           tokens_per_second=done / max(time.monotonic() - started, 1e-9))

            try:
                result = cells.write(cell_id, body.text, chunk_size=body.chunk_size, progress=progress,
                                     cancelled=lambda: job["cancel"])
                job.update(event="write_completed" if result else "write_cancelled",
                           result=cell_json(result) if result else None)
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
    @app.post("/api/read")
    async def query(body: Query):
        selection = body.cells
        if isinstance(selection, list) and any(isinstance(item, Selection) for item in selection):
            if not all(isinstance(item, Selection) for item in selection):
                raise HTTPException(400, "Do not mix Cell IDs and weighted selections")
            result = await operation(runtime._weighted_query, body.question, [item.cell_id for item in selection],
                                     [item.weight for item in selection], body.max_new_tokens, body.temperature)
        else:
            result = await operation(runtime.query, body.question, selection,
                                     max_new_tokens=body.max_new_tokens, temperature=body.temperature)
        return {**asdict(result), "answer": result.text, "cells_used": result.selected_cell_ids,
                "latency_ms": result.latency * 1000}

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy_paths", type=Path, nargs="*", help=argparse.SUPPRESS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--storage", type=Path, default=Path("minicells_data"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--dtype", choices=("auto", "fp16", "bf16", "fp32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.legacy_paths:
        if len(args.legacy_paths) != 4:
            parser.error("Use --model/--storage or four legacy positional paths")
        args.source, args.model, args.tokenizer, args.storage = args.legacy_paths
    import uvicorn

    dtype = {"float16": "fp16", "bfloat16": "bf16"}.get(args.dtype, args.dtype)
    with MiniCells.open(args.model, args.storage, dtype=dtype, source=args.source,
                        tokenizer=args.tokenizer) as runtime:
        uvicorn.run(app_for(runtime), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
