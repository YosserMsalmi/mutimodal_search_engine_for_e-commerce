# 🛍️ E-commerce Multimodal Search Engine

This project is an **AI-powered product search system** that combines **text** and **image embeddings** using **CLIP** and **SPLADE**, with **Qdrant** as the vector database.  
It provides a **FastAPI backend**, a **Gradio frontend**, and can be fully containerized with **Docker**.

---

## Features

-  **Multimodal Search:** Supports both text and image queries.
-  **Hybrid Retrieval:** Combines dense and sparse embeddings for better relevance.
-  **Qdrant Cloud Integration:** Uses Qdrant as a managed vector database.
-  **FastAPI API:** REST endpoints for searching and managing the collection.
-  **Gradio Frontend:** Simple web UI for testing and visualization.
-  **Docker Ready:** Fully containerized for deployment or local testing.

---

## !!! Important Note !!!

>  **Origin:**  
> This project is based on the notebook `Qdrant+CLIP+SPLADE.ipynb` that demonstrates hybrid (dense + sparse) multimodal retrieval.  
> The code from the notebook has been **modularized** into a clean **FastAPI application** to allow deployment, testing via API, and Dockerization.  
> You can still refer to the original notebook for reference, experimentation, or additional model explanations.

# Local Setup: 

Create a virtual environment
python -m venv venv
source venv/bin/activate    # (Windows: venv\Scripts\activate)

**Install dependencies**
pip install -r requirements.txt

**Run FastAPI backend**
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

---

# Run with Docker

**Build the image**
docker build -t ecommerce-search .

**Run the container**
docker run -p 8000:8000 -p 7860:7860 --env-file .env ecommerce-search


In both scenarios you can access:

**FastAPI API**: http://localhost:8000/docs

**Gradio app**: http://localhost:7860



