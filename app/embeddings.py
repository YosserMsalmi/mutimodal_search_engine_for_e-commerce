# app/embeddings.py
import os
from typing import List, Tuple, Any
import numpy as np
import torch
from PIL import Image
import io

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Use config or env to choose device
from .config import settings

DEVICE = settings.DEVICE

# Dense text model
dense_text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)

# CLIP text & image (sentence-transformers offers clip-ViT-B-32)
clip_model = SentenceTransformer("clip-ViT-B-32", device=DEVICE)

# SPLADE sparse model (from your notebook)
sparse_model_name = "naver/efficient-splade-VI-BT-large-doc"
tokenizer = AutoTokenizer.from_pretrained(sparse_model_name, use_fast=True)
sparse_model = AutoModelForMaskedLM.from_pretrained(sparse_model_name).to(DEVICE)
sparse_model.eval()

# -------------------------
# Dense text embeddings
# -------------------------
def embed_text_dense(texts: List[str], batch_size: int = None) -> np.ndarray:
    if batch_size is None:
        batch_size = settings.BATCH_SIZE
    emb = dense_text_model.encode(texts, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=False)
    return emb  # shape (N, dim)

# -------------------------
# CLIP text embeddings (optional extra text vector)
# -------------------------
def embed_clip_text(texts: List[str], batch_size: int = None) -> np.ndarray:
    if batch_size is None:
        batch_size = settings.BATCH_SIZE
    emb = clip_model.encode(texts, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=False)
    return emb  # shape (N, clip_dim)

# -------------------------
# CLIP image embeddings
# -------------------------
def embed_images_from_bytes(list_of_bytes: List[bytes], batch_size: int = 16) -> np.ndarray:
    imgs = []
    for b in list_of_bytes:
        try:
            img = Image.open(io.BytesIO(b)).convert("RGB")
            imgs.append(img)
        except Exception as e:
            # maybe input was a file path string in some contexts
            raise ValueError(f"Could not decode image bytes: {e}")
    emb = clip_model.encode(imgs, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=False)
    return emb  # shape (N, clip_dim)

# -------------------------
# SPLADE sparse encoding
# returns (indices_list, values_list) per document
# -------------------------
from qdrant_client.http import models as qmodels
import numpy as np

def compute_sparse_for_texts(texts: List[str], max_len: int = 256, batch_size: int = None) -> List[Tuple[List[int], List[float]]]:
    """
    Returns for each text a tuple (indices, values), where indices are token/feature ids and values are floats.
    This mirrors the notebook's approach: run sparse_model -> relu(logits).log1p() -> max-pool over tokens.
    """
    if batch_size is None:
        batch_size = settings.BATCH_SIZE

    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            logits = sparse_model(**enc).logits  # (B, L, V)
            scores = torch.relu(logits).log1p()  # non-negative
            pooled, _ = torch.max(scores, dim=1)  # (B, V)
        pooled = pooled.cpu().numpy()
        for row in pooled:
            nz = np.where(row > 0)[0]
            vals = row[nz].tolist()
            results.append((nz.tolist(), vals))
    return results

def compute_sparse_for_single(text: str, **kwargs):
    idxs_vals = compute_sparse_for_texts([text], **kwargs)[0]
    return idxs_vals

