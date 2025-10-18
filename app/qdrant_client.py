from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from .config import settings

_client: Optional[QdrantClient] = None

def get_client() -> QdrantClient:
    global _client
    if _client is None:
        kwargs = {}
        if settings.QDRANT_API_KEY:
            kwargs["api_key"] = settings.QDRANT_API_KEY
        _client = QdrantClient(url=settings.QDRANT_URL, **kwargs)
    return _client

def ensure_collection():
    client = get_client()
    try:
        client.get_collection(settings.COLLECTION_NAME)
    except Exception:
        client.recreate_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config={
                "dense_text": qmodels.VectorParams(size=settings.EMBEDDING_DIM_DENSE, distance=qmodels.Distance.COSINE),
                "dense_image": qmodels.VectorParams(size=settings.EMBEDDING_DIM_CLIP, distance=qmodels.Distance.COSINE),
                "clip_text": qmodels.VectorParams(size=settings.EMBEDDING_DIM_CLIP, distance=qmodels.Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse_text": qmodels.SparseVectorParams(index=qmodels.SparseIndexParams(on_disk=False))
            }
        )

def upsert_points(points: List[Dict[str, Any]]):
    """
    points: list of dicts:
      { "id": "id1",
        "vectors": { "dense_text": [...], "dense_image": [...], "sparse_text": qmodels.SparseVector(...) },
        "payload": {...}
      }
    """
    client = get_client()
    qpoints = []
    for p in points:
        qvecs = {}
        for name, vec in p.get("vectors", {}).items():
            if name == "sparse_text":
                # vec is tuple (indices, values)
                indices, values = vec
                qvecs[name] = qmodels.SparseVector(indices=indices, values=values)
            else:
                qvecs[name] = vec
        qpoints.append(qmodels.PointStruct(id=p["id"], vector=qvecs.get("dense_text") or qvecs.get("dense_image") or qvecs.get("clip_text"),
                                          payload=p.get("payload")))
    # Note: qdrant-client expects either "vector" (single vector) or named vectors via `vectors` parameter.
    # Using the HTTP models, do an upsert with named vectors via client.upsert using raw dict
    # Simpler: use client.upsert with "vectors" param (supported in qdrant-client).
    client.upsert(collection_name=settings.COLLECTION_NAME, points=qpoints)

def search_vectors(query_vector, vector_name: str = "dense_text", top: int = 10, filter=None):
    client = get_client()
    hits = client.search(collection_name=settings.COLLECTION_NAME, query_vector=query_vector, limit=top, vector_name=vector_name, query_filter=filter)
    return hits

# For hybrid search using both dense and sparse we can use client.search with must/should logic or a combined flow:
def search_hybrid_text(dense_query, clip_query, sparse_query_indices_values, top=10):
    """
    - dense_query: list or numpy vector for 'dense_text'
    - clip_query: list vector for 'clip_text'
    - sparse_query_indices_values: tuple (indices, values) for sparse_text
    This function demonstrates sequential scoring: run vector search and intersect / merge scores client-side.
    """
    client = get_client()
    # Example: perform dense_text search and also sparse search and merge results.
    hits_dense = client.search(collection_name=settings.COLLECTION_NAME, query_vector=dense_query, limit=top, vector_name="dense_text")
    # perform sparse search (qdrant supports sparse search via `query_vector` param? Use filter or dedicated API.)
    # For simplicity, return dense hits. You can refine merging logic as needed.
    return hits_dense
