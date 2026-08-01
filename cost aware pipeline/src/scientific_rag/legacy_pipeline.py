
import asyncio
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib
import logging
import os
import re
import shutil
import time
from typing import Any, Dict, List, Literal, Optional
import xml.etree.ElementTree as ET

# Third-party libraries
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# Bioinformatics & Arxiv
from Bio import Entrez
import arxiv

# LangChain Core & Prompts
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# LangChain Text Splitters & Retrievers
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import ArxivLoader

# LangChain Vector Stores
from langchain_chroma import Chroma  # Replaces langchain_community.vectorstores.Chroma

# LangChain Integrations (Google, OpenAI, Groq, HuggingFace)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from google import genai
from google.genai import types

# Pydantic
from pydantic import BaseModel, Field

from model_policy import ModelPolicy

from .domain import ChunkQAResponse, GatekeeperReview, QAPair, QueryPlan, ResearchPlan, SubQuery
from .pipeline_config import PipelineConfig

# --- 1. CONFIGURATION & LOGGING SETUP ---

# Load environment variables (create a .env file locally)
load_dotenv()

logger = logging.getLogger("ResearchAgent")



config = PipelineConfig()


def create_chat_model(role: str, temperature: float, model: Optional[str] = None):
    """Create a role-specific chat model with generation-safe parameters."""
    route = config.MODEL_POLICY.route_for(role)
    selected_model = model or route.primary
    kwargs = {
        "model": selected_model,
        "google_api_key": config.GOOGLE_API_KEY,
    }
    kwargs.update(config.MODEL_POLICY.generation_parameters_for_model(selected_model, temperature))
    return ChatGoogleGenerativeAI(**kwargs)


def create_generation_config(role: str, response_schema, temperature: float, model: Optional[str] = None):
    """Build Google GenAI configuration without deprecated Gemini 3 sampling fields."""
    kwargs = {
        "response_mime_type": "application/json",
        "response_schema": response_schema,
    }
    selected_model = model or config.MODEL_POLICY.route_for(role).primary
    kwargs.update(config.MODEL_POLICY.generation_parameters_for_model(selected_model, temperature))
    return types.GenerateContentConfig(**kwargs)


def is_transient_model_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in ("503", "429", "unavailable", "deadline exceeded", "timeout"))


async def ainvoke_with_fallback(role: str, temperature: float, chain_builder, payload):
    """Invoke a LangChain pipeline, retrying once with the role's fallback model on capacity errors."""
    route = config.MODEL_POLICY.route_for(role)
    models = [route.primary] + ([route.fallback] if route.fallback != route.primary else [])
    last_error = None
    for index, model in enumerate(models):
        try:
            return await chain_builder(create_chat_model(role, temperature, model)).ainvoke(payload)
        except Exception as error:
            last_error = error
            if index == len(models) - 1 or not is_transient_model_error(error):
                raise
            logger.warning("Model %s is unavailable for role %s; trying fallback %s", model, role, route.fallback)
    raise last_error


def generate_content_with_fallback(client, role: str, contents: str, response_schema, temperature: float):
    """Call the Google GenAI SDK with a role-specific fallback on capacity failures."""
    route = config.MODEL_POLICY.route_for(role)
    models = [route.primary] + ([route.fallback] if route.fallback != route.primary else [])
    for index, model in enumerate(models):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=create_generation_config(role, response_schema, temperature, model),
            )
        except Exception as error:
            if index == len(models) - 1 or not is_transient_model_error(error):
                raise
            logger.warning("Model %s is unavailable for role %s; trying fallback %s", model, role, route.fallback)

class MinerService:
    ARXIV_TIMEOUT_SECONDS = 20
    MAX_PMC_SECTIONS_PER_PAPER = 3
    MAX_PMC_SECTION_CHARS = 4_000
    _ARXIV_STOP_WORDS = {
        "about", "applications", "current", "effects", "for", "from", "into", "latest",
        "recent", "research", "the", "uses", "what", "with",
    }
    _ARXIV_GENERIC_TERMS = {
        "approach", "approaches", "code", "codes", "correction", "error", "errors",
        "method", "methods", "quantum", "study", "system", "systems", "using",
    }

    @staticmethod
    def _normalize_title(title: str) -> str:
        if not title:
            return ""
        return re.sub(r'\W+', '', title).lower()

    @classmethod
    def _arxiv_query_terms(cls, query: str) -> tuple[set[str], set[str]]:
        """Return broad terms plus the domain-specific terms that must be present."""
        terms = {
            token.lower() for token in re.findall(r"[A-Za-z0-9-]+", query)
            if len(token) > 2 and token.lower() not in cls._ARXIV_STOP_WORDS
        }
        specific_terms = terms - cls._ARXIV_GENERIC_TERMS
        return terms, specific_terms

    @staticmethod
    def _arxiv_required_phrases(query: str) -> set[str]:
        """Extract domain phrases whose individual words are too broad to trust."""
        normalized = re.sub(r"[^a-z0-9]+", " ", query.lower()).strip()
        phrases = set()
        if re.search(r"\bsurface\s+codes?\b", normalized):
            phrases.add("surface code")
        return phrases

    @staticmethod
    def _get_keywords(query: str) -> List[str]:
        # PubMed handles a short, specific query much more reliably than a long
        # natural-language question joined entirely with AND.  Keep the leading
        # domain terms (the planner puts them first) and discard question/filler
        # language before building the fielded query.
        stop_words = {
            'a', 'an', 'and', 'are', 'can', 'compare', 'do', 'does', 'evaluate',
            'explain', 'for', 'from', 'how', 'identify', 'in', 'into',
            'investigate', 'is', 'map', 'most', 'of', 'on', 'the', 'that',
            'to', 'using', 'what', 'which', 'with',
        }
        clean = re.sub(r'[^\w\s]', ' ', query).lower()
        words = clean.split()
        return [w for w in words if w not in stop_words and len(w) > 2][:5]

    @staticmethod
    def _format_pubmed_query(keywords: List[str]) -> str:
        return " AND ".join([f"{w}[tiab]" for w in keywords])

    @staticmethod
    @staticmethod
    def parse_pmc_xml(xml_text: str) -> List[Dict[str, str]]:
        print(f"DEBUG: Parsing PMC XML... (Data size: {len(xml_text)} bytes)")
        try:
            root = ET.fromstring(xml_text)
            sections = []

            def walk_sec(sec, parent=""):
                title_el = sec.find("title")
                title = title_el.text.strip() if title_el is not None and title_el.text else "Untitled"
                path = f"{parent} > {title}" if parent else title
                paragraphs = ["".join(p.itertext()).strip() for p in sec.findall("p") if "".join(p.itertext()).strip()]
                if paragraphs:
                    sections.append({"section": title, "section_path": path, "text": "\n".join(paragraphs)})
                for child in sec.findall("sec"):
                    walk_sec(child, path)

            body = root.find(".//body")
            if body is not None:
                if not body.findall("sec"):
                    paras = ["".join(p.itertext()).strip() for p in body.findall("p") if "".join(p.itertext()).strip()]
                    if paras:
                        sections.append({"section": "Full Text", "section_path": "Full Text", "text": "\n".join(paras)})
                    elif "".join(body.itertext()).strip():
                        sections.append({"section": "Full Text (Unstructured)", "section_path": "Full Text", "text": "".join(body.itertext()).strip()})
                else:
                    for sec in body.findall("sec"):
                        walk_sec(sec)

            print(f"DEBUG: Extracted {len(sections)} sections from PMC XML.")
            return sections
        except ET.ParseError as e:
            print(f"DEBUG: XML Parsing Error: {e}")
            return []

    @staticmethod
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch_pmc_full_text(pmcid: str) -> List[Dict]:
        Entrez.email = "your_email@example.com"
        print(f"DEBUG: Fetching full text for {pmcid}")
        handle = Entrez.efetch(db="pmc", id=pmcid, rettype="full", retmode="xml")
        xml_text = handle.read()
        handle.close()
        return MinerService.parse_pmc_xml(xml_text)

    @staticmethod
    def get_pmcid_from_pmid(pmid: str) -> Optional[str]:
        print(f"DEBUG: Translating PMID {pmid} to PMCID...")
        try:
            handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=pmid)
            results = Entrez.read(handle)
            handle.close()
            if results and results[0]["LinkSetDb"]:
                pmc_id = results[0]["LinkSetDb"][0]["Link"][0]["Id"]
                print(f"DEBUG: Found PMCID: PMC{pmc_id}")
                return f"PMC{pmc_id}"
        except Exception:
            print(f"DEBUG: No PMCID found for PMID {pmid}")
            return None
        return None

    @classmethod
    def select_pmc_sections(cls, sections: List[Dict]) -> List[Dict]:
        """Keep a compact, readable evidence set before costly embedding work."""
        priority_terms = ("conclusion", "result", "discussion", "introduction", "method")
        excluded_terms = ("reference", "bibliograph", "acknowledg", "supplement", "author contribution")

        def rank(section: Dict) -> tuple[int, int]:
            title = str(section.get("section", "")).lower()
            priority = next((index for index, term in enumerate(priority_terms) if term in title), len(priority_terms))
            return priority, -len(str(section.get("text", "")))

        selected = []
        for section in sorted(sections, key=rank):
            title = str(section.get("section", ""))
            text = re.sub(r"\s+", " ", str(section.get("text", ""))).strip()
            lowered_title = title.lower()
            if not text or any(term in lowered_title for term in excluded_terms):
                continue
            # XML-to-text conversion can leave LaTex preambles and equation-heavy fragments.
            latex_markers = sum(text.count(marker) for marker in ("\\usepackage", "\\begin{", "\\documentclass", "$$"))
            alphabetic_fraction = sum(char.isalpha() for char in text) / max(len(text), 1)
            if len(text) < 250 or latex_markers > 1 or alphabetic_fraction < 0.45:
                continue
            selected.append({**section, "text": text[:cls.MAX_PMC_SECTION_CHARS]})
            if len(selected) == cls.MAX_PMC_SECTIONS_PER_PAPER:
                break
        return selected

    @staticmethod
    def mine_pubmed(query: str, limit: int = 5) -> List[Document]:
        keywords = MinerService._get_keywords(query)

        if not keywords:
            return []

        search_term = MinerService._format_pubmed_query(keywords)
        print(f"DEBUG: Searching PubMed for: '{search_term}'")
        try:
            handle = Entrez.esearch(db="pubmed", term=search_term, retmax=limit, sort="relevance")
            search_results = Entrez.read(handle)
            handle.close()
            pmids = search_results.get("IdList", [])

            print(f"DEBUG: PubMed IDs found: {pmids}")
        except Exception as e:
            print(f"DEBUG: PubMed Search Failed: {e}")
            return []

        docs = []
        for pmid in pmids:
            try:
                print(f"DEBUG: Fetching Metadata for PMID {pmid}")
                handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
                records = Entrez.read(handle)
                handle.close()
                article = records["PubmedArticle"][0]["MedlineCitation"]["Article"]
                title = article.get("ArticleTitle", "No Title")
                print(f"Article Title from PUBMED:\n---{title}---")

                abstract_data = article.get("Abstract", {}).get("AbstractText", [])
                abstract = " ".join([str(part) for part in abstract_data]) if isinstance(abstract_data, list) else str(abstract_data)

                docs.append(Document(page_content=abstract, metadata={"doc_id": f"PMID:{pmid}", "title": title, "source": "pubmed", "type": "abstract"}))

                pmcid = MinerService.get_pmcid_from_pmid(pmid)
                if pmcid:
                    sections = MinerService.fetch_pmc_full_text(pmcid)
                    selected_sections = MinerService.select_pmc_sections(sections)
                    print(f"DEBUG: Retained {len(selected_sections)}/{len(sections)} readable PMC sections for PMID {pmid}")
                    for sec in selected_sections:
                        docs.append(Document(page_content=sec["text"], metadata={"doc_id": f"PMID:{pmid}", "pmcid": pmcid, "title": title, "section": sec["section"], "source": "pubmed", "type": "full_text"}))
                    time.sleep(0.35)
            except Exception as e:
                print(f"DEBUG: Error processing PMID {pmid}: {e}")
        return docs

    @staticmethod
    def mine_arxiv(query: str, limit: int = 3) -> List[Document]:
        """Return small, citable arXiv abstracts before considering any full-text download."""
        print(f"DEBUG: Searching ArXiv for: '{query}'")
        docs = []
        try:
            client = arxiv.Client(page_size=max(limit * 3, 10), num_retries=1)
            search = arxiv.Search(query=query, max_results=max(limit * 3, 10), sort_by=arxiv.SortCriterion.Relevance)
            query_terms, specific_terms = MinerService._arxiv_query_terms(query)
            required_phrases = MinerService._arxiv_required_phrases(query)
            required_overlap = min(2, len(query_terms))
            for result in client.results(search):
                title = str(result.title).strip()
                summary = str(result.summary).strip()
                candidate_text = f"{title} {summary}".lower()
                normalized_candidate = re.sub(r"[^a-z0-9]+", " ", candidate_text)
                overlap = sum(term in candidate_text for term in query_terms)
                if required_overlap and overlap < required_overlap:
                    continue
                # Broad terms such as "quantum", "error", and "correction" are
                # insufficient evidence of relevance.  A surface-code question,
                # for example, must retain "surface" in the candidate itself.
                if specific_terms and not any(term in candidate_text for term in specific_terms):
                    continue
                if any(phrase not in normalized_candidate for phrase in required_phrases):
                    continue
                entry_url = str(result.entry_id)
                arxiv_id = entry_url.rstrip("/").split("/")[-1]
                print(f"Article Title from ARXIV:\n---{title}---")
                docs.append(Document(
                    page_content=summary,
                    metadata={
                        "doc_id": f"arxiv:{arxiv_id}",
                        "title": title,
                        "source": "arxiv",
                        "source_url": entry_url,
                        "url": entry_url,
                        "type": "abstract",
                    },
                ))
                if len(docs) >= limit:
                    break
            print(f"DEBUG: ArXiv Mining complete. Total docs: {len(docs)}")
        except Exception as e:
            print(f"DEBUG: ArXiv Mining Failed: {e}")
        return docs

    @classmethod
    async def mine(cls, query: str, source: str, limit: int = 3) -> List[Document]:
        print("DEBUG: Unified Miner started for query:\n")
        print(f"'{query}' from source: '{source}'")

        raw_results = []
        if source == "pubmed":
            raw_results = await asyncio.to_thread(cls.mine_pubmed, query, limit)
        elif source == "arxiv":
            try:
                raw_results = await asyncio.wait_for(
                    asyncio.to_thread(cls.mine_arxiv, query, limit),
                    timeout=cls.ARXIV_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                print(f"DEBUG: ArXiv search timed out after {cls.ARXIV_TIMEOUT_SECONDS}s")
        else:
            print(f"WARNING: Unknown source '{source}' requested.")

        all_docs = []
        seen_titles = set()
        for doc in raw_results:
            title = doc.metadata.get("title", "")
            norm_title = cls._normalize_title(title)
            if norm_title in seen_titles:
                if any(cls._normalize_title(d.metadata.get("title")) == norm_title and d.metadata.get("source") != doc.metadata.get("source") for d in all_docs):
                    continue
            all_docs.append(doc)
            seen_titles.add(norm_title)

        print(f"DEBUG: Unified Miner [{source}] finished. Total chunks: {len(all_docs)}")
        return all_docs

    # ==========================================
    # 3. Dynamic Q&A Generation Implementation
    # ==========================================

    @staticmethod
    async def _generate_qa_for_single_doc(client: genai.Client, doc: Document, pairs_per_doc: int) -> List[Dict]:
        """Helper worker task to handle an isolated structured LLM call safely."""
        if not doc.page_content or len(doc.page_content.strip()) < 200:
            return []

        source_id = doc.metadata.get("doc_id") or doc.metadata.get("title") or "Unknown Source"
        title = doc.metadata.get("title", "Unknown Title")

        prompt = f"""
        You are an expert medical research data scientist creating a gold-standard evaluation dataset.

        Analyze the following consolidated content from the scientific document "{title}" (ID: {source_id}).
        Generate exactly {pairs_per_doc} diverse, high-quality, and challenging Question and Answer pairs.

        CRITICAL RULES:
        1. Both the question and answer must be entirely answerable from the provided text. Do not bring in outside knowledge.
        2. Avoid generic questions like "What does this section discuss?". Make them specific to the medical/scientific concepts inside.
        3. Keep the tone scientific and authoritative.

        Document Context:
        ---
        {doc.page_content}
        ---
        """

        try:
            # Wrap synchronous SDK worker target execution within an async safe block executor
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: generate_content_with_fallback(client, "qa_generator", prompt, ChunkQAResponse, 0.3)
            )

            parsed_response = response.parsed
            results = []

            if parsed_response and parsed_response.qa_pairs:
                for pair in parsed_response.qa_pairs:
                    results.append({
                        "question": pair.question,
                        "answer": pair.answer,
                        "metadata": {
                            "source_doc_id": source_id,
                            "source_title": title,
                            "source_type": doc.metadata.get("type", "mixed"),
                            "context_ground_truth": doc.page_content  # Retained specifically for RAG verification pipelines
                        }
                    })
            return results

        except Exception as e:
            print(f"DEBUG: Failed to generate Q&A for document context {source_id}: {e}")
            return []

    @classmethod
    async def create_q_a(cls, documents: List[Document], pairs_per_doc: int = 3, max_concurrent_tasks: int = 5) -> List[Dict]:
        """
        Groups fragments/chunks by document to provide complete structural context context,
        then generates a golden evaluation dataset using parallel background tasks.
        """
        if not documents:
            print("DEBUG: No documents provided for Q&A generation.")
            return []

        print(f"DEBUG: Initializing Q&A preparation across {len(documents)} document fragments...")
        client = genai.Client(api_key=config.GOOGLE_API_KEY)
        semaphore = asyncio.Semaphore(max_concurrent_tasks)

        # 1. Group fragments/abstracts cleanly back into individual source documents
        grouped_docs = defaultdict(list)
        for doc in documents:
            doc_id = doc.metadata.get("doc_id") or doc.metadata.get("title") or "Unknown_Key"
            grouped_docs[doc_id].append(doc)

        # 2. Consolidate separate structural chunks into unified contexts
        consolidated_docs = []
        for doc_id, chunks in grouped_docs.items():
            full_context = "\n\n".join([
                f"--- Section: {c.metadata.get('section', 'Main Text')} ---\n{c.page_content}"
                for c in chunks
            ])
            consolidated_docs.append(
                Document(
                    page_content=full_context,
                    metadata={
                        "doc_id": doc_id,
                        "title": chunks[0].metadata.get("title", "Unknown Title"),
                        "type": chunks[0].metadata.get("type", "consolidated")
                    }
                )
            )

        # 3. Define concurrent task executor worker closure target wrapper
        async def worker(doc: Document):
            async with semaphore:
                return await cls._generate_qa_for_single_doc(client, doc, pairs_per_doc)

        # 4. Schedule and concurrently execute the batch operations via Event Loop gather
        print(f"DEBUG: Launching parallel processing queues for {len(consolidated_docs)} distinct targets...")
        tasks = [worker(doc) for doc in consolidated_docs]
        nested_results = await asyncio.gather(*tasks)

        # Flatten structural list of lists arrays back into one master dataset array list structure
        golden_dataset = [qa for document_pairs in nested_results for qa in document_pairs]

        print(f"DEBUG: Completed target matrix. Total Synthesized Golden Dataset Q&A Pairs: {len(golden_dataset)}")
        return golden_dataset

    @staticmethod
    async def _generate_qa_for_single_doc(client: genai.Client, doc: Document, pairs_per_doc: int) -> List[Dict]:
        # Guard clause for empty/short snippets
        if not doc.page_content or len(doc.page_content.strip()) < 150:
            return []

        source_id = doc.metadata.get("doc_id") or doc.metadata.get("title") or "Unknown"
        title = doc.metadata.get("title", "Unknown Title")

        # Safe Truncation to prevent context explosion (~3000 words max per chunk analysis)
        safe_content = doc.page_content[:12000]

        prompt = f"""
        You are an expert research data scientist building an evaluation matrix.
        Analyze the following text block from "{title}" (ID: {source_id}).
        Generate exactly {pairs_per_doc} specific, challenging Question and Answer pairs.
        Both sides must be entirely grounded in this text. No extrapolation.

        Text Extract:
        {safe_content}
        """

        # Moving synchronous SDK call to an unblocked async executor threads
        loop = asyncio.get_running_loop()

        # REMOVED the blanket try/except here so you can see errors natively in your terminal console
        response = await loop.run_in_executor(
            None, lambda: generate_content_with_fallback(client, "qa_generator", prompt, ChunkQAResponse, 0.3)
        )

        results = []

        # Hardened parsing compatibility check
        if response.parsed:
            # Handle both raw model object or dict translations safely
            qa_data = response.parsed if hasattr(response.parsed, 'qa_pairs') else ChunkQAResponse.model_validate(response.parsed)

            for pair in qa_data.qa_pairs:
                results.append({
                    "question": pair.question,
                    "answer": pair.answer,
                    "metadata": {
                        "source_doc_id": source_id,
                        "source_title": title,
                        "ground_truth_context": safe_content[:400] + "..." # keep it lightweight
                    }
                })
        else:
            # Trace if response text was generated but parsing mechanism missed it
            print(f"WARNING: Model responded but layout formatting could not be natively parsed. Raw Text: {response.text[:200]}")

        return results

    @classmethod
    async def create_q_a(cls, documents: List[Document], pairs_per_doc: int = 2, max_concurrent_tasks: int = 3) -> List[Dict]:
        """Consolidates separate document segments safely, then multi-threads synthetic dataset creation."""
        if not documents:
            print("WARNING: create_q_a received an empty document list.")
            return []

        client = genai.Client(api_key=config.GOOGLE_API_KEY)
        semaphore = asyncio.Semaphore(max_concurrent_tasks)

        # Group chunks by doc source
        grouped_docs = defaultdict(list)
        for d in documents:
            key = d.metadata.get("doc_id") or d.metadata.get("title") or "Unknown"
            grouped_docs[key].append(d)

        consolidated = []
        for key, chunks in grouped_docs.items():
            # If a document has too many chunks (like your 77), we slice them to avoid overloading
            # This keeps context sizes uniform and highly punchy for evaluation pairs
            active_chunks = chunks[:4]
            full_context = "\n\n".join([f"--- Section: {c.metadata.get('section', 'Main')} ---\n{c.page_content}" for c in active_chunks])

            consolidated.append(Document(
                page_content=full_context,
                metadata={"doc_id": key, "title": chunks[0].metadata.get("title", "Unknown")}
            ))

        async def worker(doc: Document):
            async with semaphore:
                try:
                    return await cls._generate_qa_for_single_doc(client, doc, pairs_per_doc)
                except Exception as e:
                    # Let us know EXACTLY what line or API ceiling failed
                    print(f"CRITICAL ERROR generating QA for Document {doc.metadata.get('doc_id')}: {e}")
                    return []

        nested_results = await asyncio.gather(*[worker(doc) for doc in consolidated])
        final_flat_list = [item for sublist in nested_results for item in sublist]

        print(f"SUCCESS: Synthetic compilation generated {len(final_flat_list)} golden QA records.")
        return final_flat_list

class VectorService:
    def __init__(self, persist_path: str = config.CHROMA_PATH):
        self.persist_path = persist_path

        # 1. Google Gemini Embeddings
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model=config.MODEL_EMBEDDING,
            google_api_key=config.GOOGLE_API_KEY,
            task_type="retrieval_document"
        )

        # 2. Local Re-ranker
        # self.reranker = CrossEncoder(config.MODEL_CROSS_ENCODER)


        # 4. Scientific Splitters
        separators = ["\n\n", "\n", ". ", " ", ""]
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500,
            chunk_overlap=500,
            separators=separators
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=75,
            separators=separators
        )

        # 5. Storage & Retrieval States
        self.vector_store = self.init_db()
        self.parent_docstore: Dict[str, Document] = {}
        self.bm25_retriever = None

    def init_db(self):
        try:
            return Chroma(
                persist_directory=self.persist_path,
                embedding_function=self.embedding_model,
                collection_name="sci_rag_v1"
            )
        except Exception as e:
            print(f"ERROR: Failed to initialize VectorDB: {e}")
            raise

    def reset_database(self):
        if os.path.exists(self.persist_path):
            shutil.rmtree(self.persist_path)
        self.parent_docstore = {}
        self.vector_store = self.init_db()
        print("DEBUG: Database reset for new chunk sizes.")

    def _generate_id(self, doc: Document) -> str:
        source = doc.metadata.get('source', 'unknown')
        page = doc.metadata.get('page', '0')
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
        return f"{source}_p{page}_{content_hash}"

    async def add_documents(self, raw_documents: List[Document], batch_size: int = 100):
        if not raw_documents:
            return

        parents = self.parent_splitter.split_documents(raw_documents)
        child_chunks = []
        new_parent_docs = []

        for parent in parents:
            parent_id = self._generate_id(parent)
            if parent_id in self.parent_docstore:
                continue

            parent.metadata["parent_id"] = parent_id
            self.parent_docstore[parent_id] = parent
            new_parent_docs.append(parent)

            sub_docs = self.child_splitter.split_documents([parent])
            for i, child in enumerate(sub_docs):
                child.metadata["parent_id"] = parent_id
                child.metadata["child_id"] = f"{parent_id}_c{i}"
                child_chunks.append(child)

        if child_chunks:
            all_child_ids = [c.metadata["child_id"] for c in child_chunks]
            existing_data = self.vector_store.get(ids=all_child_ids)
            existing_ids = set(existing_data['ids'])

            final_children = [c for c in child_chunks if c.metadata["child_id"] not in existing_ids]
            final_ids = [c.metadata["child_id"] for c in final_children]

            if final_children:
                print(f"DEBUG: Indexing {len(final_children)} new child chunks...")
                for i in range(0, len(final_children), batch_size):
                    batch_docs = final_children[i : i + batch_size]
                    batch_ids = final_ids[i : i + batch_size]
                    print(f"DEBUG: Indexing batch {i//batch_size + 1}/{(len(final_children)-1)//batch_size + 1}...")

                    await self.vector_store.aadd_documents(documents=batch_docs, ids=batch_ids)
                    await asyncio.sleep(1.5)

        if new_parent_docs or not self.bm25_retriever:
            self.bm25_retriever = BM25Retriever.from_documents(list(self.parent_docstore.values()))
            print(f"DEBUG: BM25 updated. Total parents: {len(self.parent_docstore)}")

    # --- NEW HELPER FOR CANDIDATE FETCHING ---
    async def _get_base_candidates(self, query: str, k: int, fetch_k: int, lambda_mult: float, method: str):
        """Standardized logic for hybrid retrieval and lineage logging"""
        child_hits = await self.vector_store.amax_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
        )

        print(f"\n--- LINEAGE REPORT: {method} ---")
        vector_parent_ids = set()
        for i, c in enumerate(child_hits[:3]):
            pid = c.metadata.get("parent_id")
            if pid in self.parent_docstore:
                print(f"  Child {i} -> Found Parent: {pid}")
                vector_parent_ids.add(pid)

        loop = asyncio.get_running_loop()
        bm25_hits = await loop.run_in_executor(None, self.bm25_retriever.invoke, query) if self.bm25_retriever else []

        candidates = []
        seen_hashes = set()
        all_potential_parents = [self.parent_docstore[pid] for pid in vector_parent_ids] + bm25_hits

        for doc in all_potential_parents:
            h = hashlib.md5(doc.page_content.encode()).hexdigest()
            if h not in seen_hashes:
                candidates.append(doc)
                seen_hashes.add(h)
        return candidates

    async def retrieve_context(self, query: str, top_n: int = 5, lambda_mult: float = 0.5) -> List[Document]:
        """
        Pure Hybrid Retrieval: Combines Vector (MMR) and BM25
        without a reranker/filter to ensure maximum recall.
        """
        # 1. Get Child chunks via Vector Search (MMR)
        child_hits = await self.vector_store.amax_marginal_relevance_search(
            query,
            k=top_n * 4, # Fetch more to allow for overlap with BM25
            fetch_k=50,
            lambda_mult=lambda_mult
        )

        # 2. Get Parent IDs from those children
        vector_parent_ids = {c.metadata["parent_id"] for c in child_hits}

        # 3. Get Keyword-based hits from BM25 (Parents only)
        loop = asyncio.get_running_loop()
        bm25_hits = await loop.run_in_executor(None, self.bm25_retriever.invoke, query) if self.bm25_retriever else []

        # 4. Merge candidates and deduplicate
        candidates = []
        seen_hashes = set()

        # Prioritize Vector hits, then BM25
        all_potential_parents = [self.parent_docstore[pid] for pid in vector_parent_ids if pid in self.parent_docstore] + bm25_hits

        for doc in all_potential_parents:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if content_hash not in seen_hashes:
                candidates.append(doc)
                seen_hashes.add(content_hash)

            if len(candidates) >= top_n:
                break

        print(f"DEBUG: Retrieved {len(candidates)} documents via Hybrid Search")
        return candidates

class ResearchAgent:
    def __init__(self, vector_service, max_synthesis_subquestions: int = 2, retrieval_documents: int = 6, max_context_documents: int = 8, progress_callback=None):
        print("DEBUG: 🛠️ Initializing ResearchAgent...")
        self.vector_service = vector_service
        self.max_synthesis_subquestions = max_synthesis_subquestions
        self.retrieval_documents = retrieval_documents
        self.max_context_documents = max_context_documents
        self.progress_callback = progress_callback

        self.planner = create_chat_model("planner", 0).with_structured_output(ResearchPlan)

        self.expert = create_chat_model("expert", 0.3)

    async def _progress(self, stage: str, role: str | None = None):
        if self.progress_callback:
            await self.progress_callback(stage, role)

    async def decompose_query_async(self, query: str) -> List[str]: #[new]
        await self._progress("Breaking the question into focused research topics", "planner")
        print(f"DEBUG: 🧠 Decomposing query: '{query}'")
        prompt = ChatPromptTemplate.from_template(
            "You are a senior research scientist. Break this complex question into "
            "3-5 search-friendly sub-questions: {question}"
        )
        try:
            plan = await ainvoke_with_fallback(
                "planner",
                0,
                lambda model: prompt | model.with_structured_output(ResearchPlan),
                {"question": query},
            )
            print(f"DEBUG: ✅ Sub-questions generated: {plan.sub_questions}")
            return plan.sub_questions[:self.max_synthesis_subquestions]
        except Exception as e:
            print(f"DEBUG: ❌ Planning failed: {e}. Falling back to original query.")
            return [query]

    async def _get_docs_async(self, sub_query: str) -> List[Document]:
        print(f"DEBUG: 🔍 Hybrid Retrieval started for: '{sub_query}'")

        # 1. Directly await the async method.
        # Do NOT use run_in_executor here if retrieve_context is 'async def'
        docs = await self.vector_service.retrieve_context(
            query=sub_query,
            top_n=self.retrieval_documents
        )

        print(f"DEBUG: 📄 Found {len(docs)} high-quality docs for: '{sub_query}'")
        return docs

    async def run_research(self, query: str, retr: str = "context") -> Dict[str, Any]: #[edit]
        """
        Synthesizes the final answer using retrieved documents.
        This is called by the Orchestrator after mining is complete.
        """
        print("\nDEBUG: 🚀 Starting Research Synthesis...")
        start_time = time.time()

        # 1. Plan for retrieval (if not already planned by Orchestrator)
        sub_questions = await self.decompose_query_async(query)

        # 2. Parallel Retrieval from Vector Store
        await self._progress("Searching the evidence database", "embedding")
        print(f"DEBUG: ⚡ Running {len(sub_questions)} retrievals in parallel...")
        results = await asyncio.gather(*[self._get_docs_async(q) for q in sub_questions])

        # 3. Deduplicate Documents
        unique_docs = {}
        for docs in results:
            for d in docs:
                doc_id = hashlib.md5(f"{d.metadata.get('source', '')}{d.page_content}".encode()).hexdigest()
                unique_docs[doc_id] = d

        combined_docs = list(unique_docs.values())[:self.max_context_documents]

        # A citation identifies a source record, not an individual overlapping passage.
        # This keeps repeated passages from one paper traceable to the same paper in the UI.
        citation_ids = {}
        for d in combined_docs:
            source_key = (
                d.metadata.get("doc_id")
                or d.metadata.get("url")
                or d.metadata.get("title")
                or hashlib.md5(d.page_content.encode()).hexdigest()
            )
            citation_id = citation_ids.setdefault(str(source_key), f"S{len(citation_ids) + 1}")
            d.metadata["citation_id"] = citation_id

        # 4. Synthesize with Hybrid Knowledge
        # If no docs are found, the LLM will rely more on internal knowledge
        context_blocks = [
            f"--- [{d.metadata['citation_id']}: {d.metadata.get('title') or d.metadata.get('source') or 'Unknown Source'}] ---\n{d.page_content}"
            for d in combined_docs
        ]
        formatted_context = "\n\n".join(context_blocks) if context_blocks else "No external documents found."

        print("DEBUG: ✍️ Invoking Expert LLM for final synthesis...")
        await self._progress("Writing a clear evidence-based answer", "expert")
        synthesis_prompt = ChatPromptTemplate.from_template("""
            Answer the user's question clearly and directly for an informed non-specialist.

            You have access only to PROVIDED CONTEXT mined from academic databases.
            Treat it as the complete evidence base for this answer.

            STYLE:
            - Start with the answer, not a title, author line, or formal report heading.
            - Use plain language first; explain technical terms only when useful.
            - Use short sections or bullets only when they help readability.
            - State limitations, uncertainty, or safety concerns plainly.

            CITATION RULES:
            1. Every factual or technical claim must be directly supported by PROVIDED CONTEXT and cite its exact label, for example [S1].
            2. Do not add general background, examples, numerical claims, limitations, or recommendations from internal knowledge.
            3. If the context does not support part of the question, say "The retrieved sources do not establish this point" rather than filling the gap.
            4. If the sources provide conflicting info, mention the conflict and cite both sources.
            5. Do not invent citations or cite a source for a claim it does not support.

            PROVIDED CONTEXT:
            {context}

            QUESTION:
            {question}

            ANSWER:
        """) #[edit]

        answer = await ainvoke_with_fallback(
            "expert",
            0.3,
            lambda model: synthesis_prompt | model | StrOutputParser(),
            {"question": query, "context": formatted_context},
        )

        execution_time = time.time() - start_time
        print(f"DEBUG: ✨ Synthesis complete in {execution_time:.2f}s")

        return {
            "query": query,
            "answer": answer,
            "sub_questions": sub_questions,
            "sources_count": len(combined_docs),
            "time": f"{execution_time:.2f}s",
            "raw_docs_used": combined_docs
        }

# --- Updated Models ---

class Orchestrator:
    def __init__(self, vector_service, research_agent, llm='gemini', max_plan_steps: int = 2, papers_per_source: int = 3, progress_callback=None):
        self.vs = vector_service
        self.ra = research_agent
        self.max_plan_steps = max_plan_steps
        self.papers_per_source = papers_per_source
        self.progress_callback = progress_callback

        # 1. Initialize Planner LLM
        if llm == "gemini":
            self.planner = create_chat_model("orchestrator", 0)
        # elif llm == "gpt":
        #     self.planner = ChatOpenAI(model=config.MODEL_GRADER, openai_api_key=config.OPENAI_API_KEY, temperature=0)
        else:
            print(f"Warning: {llm} not recognized. Falling back to Gemini.")
            self.planner = create_chat_model("planner", 0)

        # 2. Initialize Gatekeeper LLM
        self.gate_keeper_llm = create_chat_model("gatekeeper", 0)

        self.parser = JsonOutputParser(pydantic_object=QueryPlan)
        self.review_parser = JsonOutputParser(pydantic_object=GatekeeperReview)

        print("DEBUG: Orchestrator initialized with Hybrid Parallel Planning.") #[edit]

    async def _progress(self, stage: str, role: str | None = None):
        if self.progress_callback:
            await self.progress_callback(stage, role)

    @staticmethod
    def enforce_source_policy(plan: QueryPlan) -> QueryPlan:
        """Make source coverage predictable after the LLM has planned the sub-questions."""
        technical_terms = ("ai", "algorithm", "computer vision", "deep learning", "machine learning", "model")
        for step in plan.plan:
            if step.source_type == "internal_knowledge":
                step.external_sources = []
                continue
            query = step.query.lower()
            technical_domain = bool({"technical", "physics", "education"}.intersection(step.domains))
            technical_query = any(term in query for term in technical_terms)
            medical_domain = "medical" in step.domains
            if medical_domain and (technical_domain or technical_query):
                step.external_sources = ["pubmed", "arxiv"]
            elif technical_domain or technical_query:
                step.external_sources = ["arxiv"]
            else:
                step.external_sources = ["pubmed"]
            print(f"DEBUG: Source policy selected {step.external_sources} for: {step.query}")
        return plan

    async def generate_plan(self, question: str) -> QueryPlan:
        """Decomposes the query into technical or general sub-questions."""
        await self._progress("Planning the research approach", "orchestrator")
        prompt = ChatPromptTemplate.from_template("""
            You are a Senior Research Architect. Break the question into 1-4 technical sub-questions.

            For each sub-query, you MUST determine the source_type:
            1. 'academic': Use for specialized physics, medical research, or complex technical data found in ArXiv/PubMed.
            2. 'internal_knowledge': Use for general definitions, popular mental models (like the Feynman Technique), or historical facts.

            Rules:
            1. If the question is simple, return exactly ONE sub-query.
            2. For 'Feynman Method', the learning technique is 'internal_knowledge'. Only use 'academic' if specifically asking for physics papers.
            3. Maximize information gain while minimizing redundant searches.

            Question: {question}
            {format_instructions}
        """) #[edit]

        try:
            result = await ainvoke_with_fallback(
                "orchestrator",
                0,
                lambda model: prompt | model | self.parser,
                {
                    "question": question,
                    "format_instructions": self.parser.get_format_instructions()
                },
            )
            parsed_plan = QueryPlan(**result)
            return QueryPlan(plan=parsed_plan.plan[:self.max_plan_steps])
        except Exception as e:
            print(f"Planning failed: {e}. Falling back to internal single query.")
            return QueryPlan(plan=[SubQuery(query=question, source_type="internal_knowledge", domains=["general"], reasoning="Fallback")]) #[edit]

    async def filter_plan(self, original_query: str, plan: QueryPlan) -> QueryPlan:
        """Evaluates and filters sub-queries in parallel to save time and cost."""
        await self._progress("Checking that the research plan is relevant", "gatekeeper")
        gate_prompt = ChatPromptTemplate.from_template("""
            You are a strict Research Auditor.
            Original Question: {original_query}

            Evaluate the following sub-query for:
            1. Redundancy: Does it repeat other sub-queries?
            2. Searchability: If marked 'academic', is it actually something found in PubMed/ArXiv?
            3. Alignment: Does it actually help answer the original question?

            Sub-Query: {sub_query}

            {format_instructions}
        """) #[edit]

        tasks = [
            ainvoke_with_fallback(
                "gatekeeper",
                0,
                lambda model: gate_prompt | model | self.review_parser,
                {
                    "original_query": original_query,
                    "sub_query": step.query,
                    "format_instructions": self.review_parser.get_format_instructions()
                },
            )
            for step in plan.plan
        ]

        evaluations = await asyncio.gather(*tasks)

        filtered_sub_queries = []
        for step, review in zip(plan.plan, evaluations):
            if review["relevance_score"] >= 0.7 and not review["should_skip"]:
                filtered_sub_queries.append(step)
            else:
                print(f"DEBUG: Gatekeeper REJECTED: {step.query} | Reason: {review['critique']}")

        if not filtered_sub_queries:
            print("DEBUG: Gatekeeper rejected all. Falling back to primary query.")
            return QueryPlan(plan=[plan.plan[0]]), evaluations

        return QueryPlan(plan=filtered_sub_queries), evaluations #[edit]

    async def ask_db(self, plan: QueryPlan, original_query: str):
        """Return a cached report only when existing evidence is sufficient and relevant."""
        await self._progress("Searching existing evidence and embeddings", "embedding")
        all_docs = []
        tasks = [self.ra._get_docs_async(step.query) for step in plan.plan]
        results = await asyncio.gather(*tasks)

        for docs in results:
            all_docs.extend(docs)
        unique_docs = {}
        for document in all_docs:
            key = document.metadata.get("doc_id") or document.metadata.get("url") or hashlib.sha256(document.page_content.encode()).hexdigest()
            unique_docs[str(key)] = document
        candidates = list(unique_docs.values())
        minimum_contexts = max(2, min(4, self.ra.retrieval_documents))
        diagnostics = {
            "checked": True,
            "candidate_contexts": len(candidates),
            "minimum_contexts": minimum_contexts,
            "decision": "mine_fresh",
            "reason": "insufficient_existing_evidence",
        }
        if len(candidates) < minimum_contexts:
            print(f"DEBUG: Internal DB has only {len(candidates)} usable contexts. Proceeding to mining.")
            return None, diagnostics

        verification_prompt = ChatPromptTemplate.from_template("""
            Decide whether the retrieved evidence is sufficient and directly relevant to answer the user's question.
            Reuse it only when it supports a high-quality answer without fresh source mining.

            Question: {question}
            Retrieved evidence:
            {context}

            {format_instructions}
        """)
        sample_context = "\n\n".join(document.page_content[:500] for document in candidates[:6])
        try:
            review = await ainvoke_with_fallback(
                "gatekeeper",
                0,
                lambda model: verification_prompt | model | self.review_parser,
                {
                    "question": original_query,
                    "context": sample_context,
                    "format_instructions": self.review_parser.get_format_instructions(),
                },
            )
            relevance_score = float(review["relevance_score"])
            diagnostics["relevance_score"] = relevance_score
            if relevance_score >= 0.85 and not review["should_skip"]:
                diagnostics.update({"decision": "reuse_existing", "reason": "sufficient_relevant_evidence"})
                print(f"DEBUG: Reusing {len(candidates)} relevant existing contexts (score={relevance_score:.2f}).")
                return await self.ra.run_research(original_query), diagnostics
            diagnostics["reason"] = "existing_evidence_not_relevant_enough"
            print(f"DEBUG: Existing evidence score {relevance_score:.2f} did not meet reuse threshold. Proceeding to mining.")
        except Exception as error:
            diagnostics.update({"reason": "cache_verification_failed", "error": str(error)})
            print("DEBUG: Cache verification failed; proceeding to mining for safety.")
        return None, diagnostics

    async def run_mining_operation(self, plan: QueryPlan, limit=None):
        """Executes source mining concurrently while isolating individual source failures.""" #[edit]
        await self._progress("Collecting relevant academic sources")
        all_mined_docs = []
        source_diagnostics = {}
        limit = limit or self.papers_per_source
        for step in plan.plan:
            if step.source_type == "internal_knowledge": #[new]
                print(f"DEBUG: Skipping Scraper for internal sub-query: {step.query}") #[new]
                continue #[new]

            print(f"DEBUG: Mining Academic Source: {step.query}")
            sources = step.external_sources
            for source in sources:
                source_diagnostics.setdefault(source, {"requested": 0, "collected": 0, "errors": []})["requested"] += 1
            mined = await asyncio.gather(
                *(MinerService.mine(step.query, source=source, limit=limit) for source in sources),
                return_exceptions=True,
            )
            for source, result in zip(sources, mined):
                if isinstance(result, Exception):
                    print(f"Error mining {source}: {result}")
                    source_diagnostics[source]["errors"].append(str(result))
                    continue
                source_diagnostics[source]["collected"] += len(result)
                all_mined_docs.extend(result)

        return all_mined_docs, len(all_mined_docs), source_diagnostics

    async def clear_knowledge_base(self):
        print("DEBUG: Clearing Vector Store...")
        try:
            self.vs.vector_store.delete_collection()
            self.vs.vector_store = self.vs.init_db()
        except Exception as e:
            print(f"DEBUG: Note - Could not clear collection: {e}")

    async def populate_knowledge_base(self, documents: List):
        await self._progress("Indexing sources for semantic and full-text search", "embedding")
        print(f"DEBUG: Indexing {len(documents)} documents into Vector Store...")
        await self.vs.add_documents(documents)
        self.vs.restrict_final_retrieval_to_current_run()

    async def synthesize_final_report(self, original_query: str):
        await self._progress("Preparing your answer from the evidence", "expert")
        print("DEBUG: Handing off to Research Agent for final synthesis...")
        # Research Agent uses both Vector Store and Internal LLM Knowledge
        return await self.ra.run_research(original_query)

    async def execute_full_study(self, original_query: str, clear_db: bool = False, generate_evaluation_dataset: bool = False, skip_database_cache: bool = False):
        """Main Orchestration Pipeline."""
        print(f"\n--- [ORCHESTRATOR] STARTING HYBRID STUDY ---")

        if clear_db:
            await self.clear_knowledge_base()

        # 1. Plan
        raw_plan = await self.generate_plan(original_query)

        # 2. Filter
        refined_plan, evals = await self.filter_plan(original_query, raw_plan)
        refined_plan = self.enforce_source_policy(refined_plan)

        # 3. DB Check
        cache_diagnostics = {"checked": False, "decision": "mine_fresh", "reason": "cache_bypassed"}
        existing_report = None
        if not skip_database_cache:
            existing_report, cache_diagnostics = await self.ask_db(refined_plan, original_query)
        if existing_report:
            return {"report": existing_report, "status": "retrieved_from_cache", "metadata": {"cache": cache_diagnostics}}

        # 4. Mine (Only academic parts)
        docs, doc_count, source_diagnostics = await self.run_mining_operation(refined_plan)

        # 5. Index & Synthesize
        if doc_count > 0:
            await self.populate_knowledge_base(docs)

        # Final synthesis handles both the mined docs and internal explanations
        final_report = await self.synthesize_final_report(original_query)

        golden_set = []
        if generate_evaluation_dataset and doc_count > 0:
            print("DEBUG: Asking MinerService to compile synthetic evaluation criteria...")
            golden_set = await MinerService.create_q_a(documents=docs, pairs_per_doc=5)

        return {
            "report": final_report,
            "metadata": {
                "mined_count": doc_count,
                "plan_executed": [s.query for s in refined_plan.plan],
                "source_diagnostics": source_diagnostics,
                "cache": cache_diagnostics,
            },
            "evaluation_dataset": golden_set
        }
