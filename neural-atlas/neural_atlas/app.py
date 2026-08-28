from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .model_runtime import ModelRuntime
from .safetensors_index import (
    SafetensorsIndexError,
    TileTooLargeError,
    UnsupportedDtypeError,
)


class ModelLoadRequest(BaseModel):
    model_id: str | None = Field(default=None, alias="modelId")


class TraceRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20_000)
    max_new_tokens: int = Field(default=16, alias="maxNewTokens", ge=1, le=128)
    temperature: float = Field(default=0.0, ge=0.0, le=5.0)
    top_p: float = Field(default=0.95, alias="topP", gt=0.0, le=1.0)
    capture_attention: bool = Field(default=True, alias="captureAttention")
    capture_logit_lens: bool = Field(default=True, alias="captureLogitLens")
    seed: int = 0
    session_id: str | None = Field(default=None, alias="sessionId")


class ContributionRequest(BaseModel):
    session_id: str = Field(alias="sessionId")
    generation_step: int = Field(alias="generationStep", ge=0)
    layer: int = Field(ge=0)
    output_index: int = Field(alias="outputIndex", ge=0)
    top_k: int = Field(default=32, alias="topK", ge=1, le=128)


runtime = ModelRuntime()
web_dist = Path(__file__).resolve().parents[1] / "web" / "dist"
max_tile_values = int(os.getenv("NEURAL_ATLAS_MAX_TILE_VALUES", "10000000"))

app = FastAPI(
    title="Neural Atlas API",
    version=__version__,
    description=(
        "Exact safetensors microscopy and token-by-token functional imaging "
        "for open-weight transformers."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5174",
        "http://localhost:5174",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

if (web_dist / "assets").exists():
    app.mount("/assets", StaticFiles(directory=web_dist / "assets"), name="assets")


def _http_error(error: Exception) -> HTTPException:
    if isinstance(error, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(error))
    if isinstance(error, KeyError):
        return HTTPException(status_code=404, detail=str(error))
    if isinstance(error, TileTooLargeError):
        return HTTPException(status_code=413, detail=str(error))
    if isinstance(error, (IndexError, ValueError, UnsupportedDtypeError)):
        return HTTPException(status_code=422, detail=str(error))
    if isinstance(error, SafetensorsIndexError):
        return HTTPException(status_code=500, detail=str(error))
    return HTTPException(status_code=500, detail=f"{type(error).__name__}: {error}")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "version": __version__,
        "runtime": runtime.status(),
        "frontendBuilt": (web_dist / "index.html").exists(),
        "exactWeightAtlas": True,
        "liveTrace": True,
    }


@app.get("/api/model")
async def model_metadata() -> dict[str, Any]:
    return runtime.architecture()


@app.post("/api/model/load")
async def load_model(request: ModelLoadRequest) -> dict[str, Any]:
    try:
        architecture = await runtime.load(request.model_id)
        index = await runtime.ensure_index()
        return {"architecture": architecture, "checkpoint": index.manifest()}
    except Exception as error:
        raise _http_error(error) from error


@app.post("/api/checkpoint/index")
async def index_checkpoint() -> dict[str, Any]:
    try:
        index = await runtime.ensure_index()
        return index.manifest()
    except Exception as error:
        raise _http_error(error) from error


@app.get("/api/tensors")
async def tensors(
    q: str = "",
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    try:
        index = await runtime.ensure_index()
        return index.search(q, offset=offset, limit=limit)
    except Exception as error:
        raise _http_error(error) from error


@app.get("/api/tensor")
async def tensor(name: str) -> dict[str, Any]:
    try:
        index = await runtime.ensure_index()
        return index.describe(name)
    except Exception as error:
        raise _http_error(error) from error


@app.get("/api/tile")
async def tile(
    name: str,
    row_start: int = Query(default=0, alias="rowStart", ge=0),
    column_start: int = Query(default=0, alias="columnStart", ge=0),
    row_count: int | None = Query(default=None, alias="rowCount", ge=1),
    column_count: int | None = Query(default=None, alias="columnCount", ge=1),
    target_rows: int = Query(default=64, alias="targetRows", ge=1, le=256),
    target_columns: int = Query(default=64, alias="targetColumns", ge=1, le=256),
) -> dict[str, Any]:
    try:
        index = await runtime.ensure_index()
        return await asyncio.to_thread(
            index.tile,
            name,
            row_start=row_start,
            column_start=column_start,
            row_count=row_count,
            column_count=column_count,
            target_rows=target_rows,
            target_columns=target_columns,
            max_source_values=max_tile_values,
        )
    except Exception as error:
        raise _http_error(error) from error


@app.get("/api/scalar")
async def scalar(
    name: str,
    indices: str | None = None,
    matrix_row: int | None = Query(default=None, alias="matrixRow", ge=0),
    matrix_column: int | None = Query(default=None, alias="matrixColumn", ge=0),
) -> dict[str, Any]:
    try:
        index = await runtime.ensure_index()
        if matrix_row is not None or matrix_column is not None:
            if matrix_row is None or matrix_column is None:
                raise ValueError("matrixRow and matrixColumn must be supplied together")
            return index.read_matrix_scalar(name, matrix_row, matrix_column)
        if indices is None:
            raise ValueError("Provide comma-separated indices or a matrix coordinate")
        parsed = [] if indices.strip() == "" else [int(value.strip()) for value in indices.split(",")]
        return index.read_scalar(name, parsed)
    except Exception as error:
        raise _http_error(error) from error


@app.post("/api/contribution")
async def contribution(request: ContributionRequest) -> dict[str, Any]:
    try:
        return await runtime.contribution(
            session_id=request.session_id,
            generation_step=request.generation_step,
            layer=request.layer,
            output_index=request.output_index,
            top_k=request.top_k,
        )
    except Exception as error:
        raise _http_error(error) from error


@app.websocket("/api/trace/ws")
async def trace_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        payload = await websocket.receive_json()
        request = TraceRequest.model_validate(payload)
        await websocket.send_json(
            {
                "type": "model_loading" if not runtime.loaded else "model_ready",
                "modelId": runtime.model_id,
                "device": str(runtime.device),
            }
        )
        async for event in runtime.stream_trace(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            capture_attention=request.capture_attention,
            capture_logit_lens=request.capture_logit_lens,
            seed=request.seed,
            session_id=request.session_id,
        ):
            await websocket.send_json(event)
    except WebSocketDisconnect:
        return
    except Exception as error:
        await websocket.send_json(
            {
                "type": "error",
                "error": type(error).__name__,
                "message": str(error),
            }
        )
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass


@app.get("/", include_in_schema=False)
async def frontend_root() -> Any:
    index = web_dist / "index.html"
    if index.exists():
        return FileResponse(index)
    return JSONResponse(
        {
            "name": "Neural Atlas",
            "api": "/docs",
            "message": "Build the frontend with `cd web && npm ci && npm run build`.",
        }
    )


@app.get("/{path:path}", include_in_schema=False)
async def frontend_fallback(path: str) -> Any:
    requested = (web_dist / path).resolve()
    try:
        requested.relative_to(web_dist.resolve())
    except ValueError:
        raise HTTPException(status_code=404, detail="Not found")
    if requested.is_file():
        return FileResponse(requested)
    index = web_dist / "index.html"
    if index.exists():
        return FileResponse(index)
    raise HTTPException(status_code=404, detail="Frontend has not been built")


def run() -> None:
    uvicorn.run(
        "neural_atlas.app:app",
        host=os.getenv("NEURAL_ATLAS_HOST", "127.0.0.1"),
        port=int(os.getenv("NEURAL_ATLAS_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    run()
