# app/api.py
from fastapi import APIRouter, UploadFile, File, Form
from typing import List
import numpy as np

from .embeddings import embed_text_dense, embed_clip_text, embed_images_from_bytes, compute_sparse_for_texts
from .qdrant_client import upsert_points, search_vectors
from .config import settings

router = APIRouter()

@router.post("/add_text")
async def add_text_endpoint(texts: List[str]):
    dense = embed_text_dense(texts)
    clip = embed_clip_text(texts)
    sparse = compute_sparse_for_texts(texts)

    points = []
    for i, text in enumerate(texts):
        points.append({
            "id": f"text-{i}",
            "vectors": {
                "dense_text": dense[i].tolist(),
                "clip_text": clip[i].tolist(),
                "sparse_text": sparse[i],
            },
            "payload": {"text": text},
        })
    upsert_points(points)
    return {"status": "ok", "count": len(points)}

@router.post("/search_text")
async def search_text_endpoint(query: str, top_k: int = 5):
    q_vec = embed_text_dense([query])[0]
    hits = search_vectors(q_vec, "dense_text", top=top_k)
    return {"results": [hit.payload for hit in hits]}

@router.post("/add_image")
async def add_image_endpoint(files: List[UploadFile] = File(...)):
    file_bytes = [await f.read() for f in files]
    embs = embed_images_from_bytes(file_bytes)
    points = []
    for i, emb in enumerate(embs):
        points.append({
            "id": f"img-{i}",
            "vectors": {"dense_image": emb.tolist()},
            "payload": {"filename": files[i].filename},
        })
    upsert_points(points)
    return {"status": "ok", "count": len(points)}

@router.post("/search_image")
async def search_image_endpoint(file: UploadFile = File(...), top_k: int = 5):
    emb = embed_images_from_bytes([await file.read()])[0]
    hits = search_vectors(emb, "dense_image", top=top_k)
    return {"results": [hit.payload for hit in hits]}
