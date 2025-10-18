import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Qdrant configuration
    QDRANT_URL: str
    QDRANT_API_KEY: str

        # Models
    model_dense: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_clip: str = "clip-ViT-B-32"
    model_splade: str = "naver/efficient-splade-VI-BT-large-doc"

    # Collection name and vector dimensions
    COLLECTION_NAME: str = "products_hybrid"
    EMBEDDING_DIM_DENSE: int = 384
    EMBEDDING_DIM_CLIP: int = 512

    # Embedding configuration
    DEVICE: str = "cpu"  # or "cuda" if you have GPU support
    BATCH_SIZE: int = 16  # adjust for performance

     # Server config
    host: str = "0.0.0.0"
    port: int = 8000
    gradio_port: int = 7860

    class Config:
        env_file = ".env"  # will read from .env file

settings = Settings()