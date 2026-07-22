"""Scientific RAG: a local, Docker-friendly research assistant."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, Literal

import arxiv
import gradio as gr
from Bio import Entrez, Medline
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("scientific_rag")


@dataclass(frozen=True)
class AppConfig:
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    entrez_email: str = os.getenv("ENTREZ_EMAIL", "")
    chroma_path: str = os.getenv("CHROMA_PATH", "./data/chroma")
    collection_name: str = os.getenv("CHROMA_COLLECTION", "scientific_rag")
    chat_model: str = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

    def validate(self) -> None:
        missing = []
        if not self.google_api_key:
            missing.append("GOOGLE_API_KEY")
        if not self.entrez_email:
            missing.append("ENTREZ_EMAIL")
        if missing:
            raise ValueError(f"Missing required environment variable(s): {', '.join(missing)}")


class MinerService:
    """Fetches research abstracts from public scholarly APIs."""

    def __init__(self, email: str) -> None:
        Entrez.email = email

    def search_pubmed(self, query: str, limit: int) -> list[Document]:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=limit, sort="relevance")
        result = Entrez.read(handle)
        handle.close()
        identifiers = result.get("IdList", [])
        if not identifiers:
            return []

        handle = Entrez.efetch(db="pubmed", id=",".join(identifiers), rettype="medline", retmode="text")
        records = list(Medline.parse(handle))
        handle.close()
        documents = []
        for record in records:
            abstract = record.get("AB", "").strip()
            if not abstract:
                continue
            pmid = record.get("PMID", "unknown")
            documents.append(
                Document(
                    page_content=abstract,
                    metadata={
                        "source": "pubmed",
                        "id": f"PMID:{pmid}",
                        "title": record.get("TI", "Untitled"),
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    },
                )
            )
        return documents

    def search_arxiv(self, query: str, limit: int) -> list[Document]:
        search = arxiv.Search(query=query, max_results=limit, sort_by=arxiv.SortCriterion.Relevance)
        documents = []
        for paper in arxiv.Client().results(search):
            documents.append(
                Document(
                    page_content=paper.summary,
                    metadata={
                        "source": "arxiv",
                        "id": paper.entry_id,
                        "title": paper.title,
                        "url": paper.entry_id,
                    },
                )
            )
        return documents

    def search(self, query: str, source: Literal["pubmed", "arxiv", "both"], limit: int) -> list[Document]:
        documents: list[Document] = []
        if source in ("pubmed", "both"):
            documents.extend(self.search_pubmed(query, limit))
        if source in ("arxiv", "both"):
            documents.extend(self.search_arxiv(query, limit))
        return documents


class ResearchService:
    def __init__(self, config: AppConfig) -> None:
        config.validate()
        self.config = config
        self.miner = MinerService(config.entrez_email)
        embeddings = GoogleGenerativeAIEmbeddings(
            model=config.embedding_model,
            google_api_key=config.google_api_key,
        )
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            persist_directory=config.chroma_path,
            embedding_function=embeddings,
        )
        self.llm = ChatGoogleGenerativeAI(
            model=config.chat_model,
            google_api_key=config.google_api_key,
            temperature=0.2,
        )

    def research(
        self,
        query: str,
        source: Literal["pubmed", "arxiv", "both"],
        limit: int,
        progress: Callable[[float, str], None] | None = None,
    ) -> tuple[str, str]:
        if not query.strip():
            return "Please enter a research question.", ""
        if progress:
            progress(0.15, "Searching scholarly sources…")
        documents = self.miner.search(query, source, limit)
        if not documents:
            return "No abstracts were found. Try a broader query.", ""

        if progress:
            progress(0.45, f"Indexing {len(documents)} abstracts…")
        self.vector_store.add_documents(documents)
        if progress:
            progress(0.65, "Retrieving the most relevant evidence…")
        context_documents = self.vector_store.similarity_search(query, k=min(6, len(documents)))
        context = "\n\n".join(
            f"[{index}] {document.metadata.get('title', 'Untitled')}\n{document.page_content}"
            for index, document in enumerate(context_documents, start=1)
        )
        prompt = ChatPromptTemplate.from_template(
            "Answer the research question using only the supplied abstracts. Be clear about uncertainty, "
            "do not give medical advice, and cite claims using bracket numbers such as [1].\n\n"
            "Question: {question}\n\nAbstracts:\n{context}"
        )
        if progress:
            progress(0.8, "Writing an evidence-based answer…")
        answer = (prompt | self.llm).invoke({"question": query, "context": context}).content
        sources = "\n".join(
            f"- [{index}] [{document.metadata.get('title', 'Untitled')}]({document.metadata.get('url', '')})"
            for index, document in enumerate(context_documents, start=1)
        )
        return str(answer), sources


def create_app() -> gr.Blocks:
    config = AppConfig()
    service: ResearchService | None = None

    def run(query: str, source: str, limit: int, progress=gr.Progress()) -> tuple[str, str]:
        nonlocal service
        try:
            progress(0.05, "Preparing the research service…")
            service = service or ResearchService(config)
            answer, sources = service.research(query, source, int(limit), progress)
            progress(1.0, "Done")
            return answer, sources
        except Exception as error:
            logger.exception("Research request failed")
            return f"Request failed: {error}", ""

    with gr.Blocks(title="Scientific RAG") as app:
        gr.Markdown("# Scientific RAG\nSearch scholarly abstracts and answer questions with cited context.")
        with gr.Row():
            query = gr.Textbox(label="Research question", placeholder="What is known about…", scale=4)
            source = gr.Dropdown(["both", "pubmed", "arxiv"], value="both", label="Sources", scale=1)
            limit = gr.Slider(1, 10, value=5, step=1, label="Results per source", scale=1)
        submit = gr.Button("Research", variant="primary")
        answer = gr.Markdown(label="Answer")
        sources = gr.Markdown(label="Retrieved sources")
        submit.click(run, inputs=[query, source, limit], outputs=[answer, sources])
        query.submit(run, inputs=[query, source, limit], outputs=[answer, sources])
    return app


if __name__ == "__main__":
    create_app().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
    )
