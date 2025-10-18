# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import router as api_router
from .qdrant_client import ensure_collection
from .gradio_ui import build_interface

import gradio as gr

app = FastAPI(title="Multimodal Search API")

# Allow frontend testing from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Qdrant
ensure_collection()

# Include API routes
app.include_router(api_router, prefix="/api")

# Mount Gradio UI
demo = build_interface()
app = gr.mount_gradio_app(app, demo, path="/")

@app.get("/")
async def root():
    return {"message": "Multimodal API running", "docs": "/docs"}
