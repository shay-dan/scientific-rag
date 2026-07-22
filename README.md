# Scientific RAG

Scientific research assistant that searches PubMed and arXiv, stores abstracts in Chroma, and answers questions using Gemini with retrieved source citations.

## Configuration

Copy `.env.example` to `.env` and set `GOOGLE_API_KEY` and `ENTREZ_EMAIL`.

## Run locally

Create a virtual environment, install dependencies, then run `python scientific_rag.py`. The interface is available on port 7860.

## Run with Docker

Copy `.env.example` to `.env`, set the required values, then run `docker compose up --build`. Persistent Chroma data is stored in the `chroma_data` Docker volume.
