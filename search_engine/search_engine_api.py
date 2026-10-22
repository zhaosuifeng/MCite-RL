"""Configurable Wiki-VISA retrieval API; model loading happens at startup."""
from __future__ import annotations

try:
    from .paths import project_path, relative_path, model_reference
except ImportError:
    from paths import project_path, relative_path, model_reference

import argparse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def create_app(index_dir: str | Path, image_root: str | Path,
               model_name: str, engine_factory: Callable | None = None):
    from fastapi import FastAPI, HTTPException, Query

    index_dir = project_path(index_dir)
    image_root = project_path(image_root)

    @asynccontextmanager
    async def lifespan(app):
        if engine_factory is None:
            if not index_dir.is_dir() or not any(index_dir.glob("*.node")):
                raise RuntimeError(f"No .node index files in {index_dir}")
            if not image_root.is_dir():
                raise RuntimeError(f"Image directory does not exist: {image_root}")
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("Local retrieval needs a CUDA GPU; use a remote search service until a GPU is available")
            if __package__:
                from .search_engine import SearchEngine
            else:
                from search_engine import SearchEngine
            engine = SearchEngine(dataset_dir=str(index_dir.parent),
                                  node_dir_prefix=index_dir.name, embed_model_name=model_name)
        else:
            engine = engine_factory()
        app.state.engine = engine
        image_index: dict[str, Path] = {}
        for path in sorted(image_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                image_index.setdefault(path.stem, path)
        app.state.image_index = image_index
        yield

    app = FastAPI(title="Wiki-VISA Retrieval", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "images": len(app.state.image_index)}

    @app.get("/search")
    def search(queries: list[str] = Query(...)) -> list[list[dict[str, Any]]]:
        if not queries or any(not q.strip() for q in queries):
            raise HTTPException(status_code=400, detail="Queries must not be empty")
        batches = app.state.engine.batch_search(queries)
        results = []
        for candidates in batches:
            items = []
            for rank, file_name in enumerate(candidates):
                image_id = Path(str(file_name)).stem
                path = app.state.image_index.get(image_id)
                if path is None:
                    raise HTTPException(status_code=409, detail=f"Index/image mismatch: no image for {image_id}")
                items.append({"idx": rank, "image_id": image_id, "image_file": relative_path(path)})
            results.append(items)
        # Keep the existing bare-list response used by mcite_agent and VERL.
        return results

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index_dir", default="corpus/wiki-visa/val_batch_0")
    parser.add_argument("--image_root", default="data/wiki-visa/images")
    parser.add_argument("--embed_model_name", default="vidore/colqwen2-v1.0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(args.index_dir, args.image_root, args.embed_model_name),
                host=args.host, port=args.port)


if __name__ == "__main__":
    main()
