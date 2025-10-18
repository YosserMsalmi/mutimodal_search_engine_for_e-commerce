# app/gradio_ui.py
import gradio as gr
from .embeddings import embed_text_dense
from .qdrant_client import search_vectors

def search_text(query, top_k=5):
    q_vec = embed_text_dense([query])[0]
    hits = search_vectors(q_vec, "dense_text", top=top_k)
    return "\n".join([str(hit.payload) for hit in hits])

def build_interface():
    with gr.Blocks() as demo:
        gr.Markdown("### 🔍 Multimodal Search Demo (Text)")
        query = gr.Textbox(label="Enter your query:")
        k = gr.Slider(1, 10, value=5, step=1, label="Top K results")
        btn = gr.Button("Search")
        output = gr.Textbox(label="Results")
        btn.click(fn=search_text, inputs=[query, k], outputs=output)
    return demo
