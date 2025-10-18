import io
import numpy as np
from PIL import Image
from .qdrant_client import client, settings
from .embeddings import (
    embed_text_dense,
    embed_clip_text,
    embed_images_from_bytes,
)
from .config import settings

def final_text(query: str):
    """You can add correction/translation logic here later."""
    return {"query_corrected": query, "query_translated": query}

# --------------- Text Search ----------------
def search_by_text(query: str, top_k=6):
    # Hybrid: dense + CLIP text
    dense_vec = embed_text_dense([query])[0]
    clip_vec = embed_clip_text([query])[0]
    combined_vec = np.concatenate([dense_vec, clip_vec])

    hits = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=combined_vec.tolist(),
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "id": h.id,
            "name": h.payload.get("name", ""),
            "score": h.score,
            "image": h.payload.get("image"),
        }
        for h in hits
    ]

# --------------- Image Search ----------------
def search_by_image(image_path: str, top_k=6):
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_emb = embed_images_from_bytes([img_bytes])[0]

    hits = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=img_emb.tolist(),
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "id": h.id,
            "name": h.payload.get("name", ""),
            "score": h.score,
            "image": h.payload.get("image"),
        }
        for h in hits
    ]
