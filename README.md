---
title: Scientific RAG
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
short_description: Search scholarly abstracts and generate cited answers.
---

# Scientific RAG

Scientific research assistant that plans and validates sub-queries, mines PubMed and arXiv, performs hybrid retrieval, and synthesizes a cited report with Gemini.

## Configuration

Copy `.env.example` to `.env` and set `GOOGLE_API_KEY` and `ENTREZ_EMAIL`.

The model settings are role-based. The default production policy uses Gemini 3.6 Flash for orchestration and final synthesis, Gemini 3.5 Flash-Lite for planning, gatekeeping, and QA generation, and compatible Gemini 2.5 fallbacks when a primary model returns a transient capacity error.

## Run locally

Create a virtual environment, install dependencies, then run `python production_app.py`. The interface is available on port 7860.

The original pipeline is preserved in `new_scientific_rag.py`. The production application package lives in `src/scientific_rag/`; it currently calls `Orchestrator.execute_full_study` and therefore uses the planner, gatekeeper, miner, hybrid vector/BM25 retrieval, synthesizer, and optional synthetic QA generation.

## Run with Docker

Copy `.env.example` to `.env`, set the required values, then run `docker compose up --build`. Persistent Chroma data is stored in the `chroma_data` Docker volume.

## Deploy to Hugging Face Spaces

1. Create a new Hugging Face Space and select **Docker** as its SDK.
2. In the Space's **Settings**, add `GOOGLE_API_KEY` as a **Secret** and `ENTREZ_EMAIL` as a **Variable**.
3. Add the Space as a Git remote, then push this repository:

   ```powershell
   git remote add huggingface https://huggingface.co/spaces/YOUR-HF-USERNAME/scientific-rag.git
   git push huggingface main
   ```

The free runtime may restart or sleep, which clears the local Chroma index. The app automatically rebuilds its index as users search.
