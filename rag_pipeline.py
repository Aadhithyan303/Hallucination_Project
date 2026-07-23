"""
Self-Verifying RAG System — RAG Pipeline (Research-Grade)
==========================================================
Improvements over v1:
  • TXT, PDF, and directory document loaders
  • Larger chunks (800 chars) with 150-char overlap for better recall
  • FAISS index persistence (save/load from disk)
  • Unified `ingest()` entry point accepting texts, files, or directories
"""

import os
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class RetrievedContext:
    """Holds retrieved documents and metadata for one query."""
    query: str
    documents: List[Document]
    scores: List[float] = field(default_factory=list)

    def to_context_string(self) -> str:
        """Concatenate document content into a single plain context string."""
        return "\n\n".join(doc.page_content for doc in self.documents)


# ─────────────────────────────────────────────
# Document Processor
# ─────────────────────────────────────────────

class DocumentProcessor:
    """
    Loads and chunks documents from various sources.

    Args:
        chunk_size:    Max characters per chunk (default 800)
        chunk_overlap: Overlap between consecutive chunks (default 150)
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ── Raw text ──────────────────────────────────────────────────────────────

    def from_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
    ) -> List[Document]:
        """Create Document objects from raw text strings."""
        if metadatas is None:
            metadatas = [{"source": f"text_{i}"} for i in range(len(texts))]
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        return self.splitter.split_documents(docs)

    # ── Single file ───────────────────────────────────────────────────────────

    def from_file(self, filepath: str) -> List[Document]:
        """Load and chunk a single .txt or .pdf file."""
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(filepath)
        else:
            loader = TextLoader(filepath, encoding="utf-8")
        raw = loader.load()
        chunks = self.splitter.split_documents(raw)
        print(f"[Loader] {filepath} → {len(chunks)} chunks")
        return chunks

    # ── Directory ─────────────────────────────────────────────────────────────

    def from_directory(self, dirpath: str, glob: str = "**/*.txt") -> List[Document]:
        """
        Load all matching files from a directory.
        Use glob='**/*.pdf' for PDFs, or '**/*' for everything.
        """
        loader = DirectoryLoader(dirpath, glob=glob)
        raw = loader.load()
        chunks = self.splitter.split_documents(raw)
        print(f"[Loader] {dirpath} ({glob}) → {len(raw)} docs → {len(chunks)} chunks")
        return chunks


# ─────────────────────────────────────────────
# Vector Store Manager
# ─────────────────────────────────────────────

class VectorStoreManager:
    """
    Manages a FAISS vector store: build, merge, save, load, and retrieve.

    Args:
        model_name: HuggingFace sentence-transformer model for embeddings.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        print(f"[VectorStore] Loading embedding model: {model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore: Optional[FAISS] = None

    def build(self, documents: List[Document]) -> None:
        """Build a fresh FAISS index from a list of Document chunks."""
        print(f"[VectorStore] Building index from {len(documents)} chunks...")
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        print("[VectorStore] Index built.")

    def add(self, documents: List[Document]) -> None:
        """Merge new chunks into an existing index (or build if empty)."""
        if self.vectorstore is None:
            self.build(documents)
        else:
            self.vectorstore.add_documents(documents)
            print(f"[VectorStore] Added {len(documents)} chunks to existing index.")

    def save(self, path: str = "faiss_index") -> None:
        """Persist the FAISS index to disk."""
        if self.vectorstore is None:
            raise ValueError("No index to save. Call build() first.")
        self.vectorstore.save_local(path)
        print(f"[VectorStore] Index saved to '{path}'")

    def load(self, path: str = "faiss_index") -> None:
        """Load a persisted FAISS index from disk."""
        self.vectorstore = FAISS.load_local(
            path, self.embeddings, allow_dangerous_deserialization=True
        )
        print(f"[VectorStore] Index loaded from '{path}'")

    def retrieve(self, query: str, top_k: int = 5) -> RetrievedContext:
        """Retrieve top-k most relevant chunks for a query."""
        if self.vectorstore is None:
            raise ValueError("Index not initialised. Call build() or load() first.")

        results: List[Tuple[Document, float]] = \
            self.vectorstore.similarity_search_with_score(query, k=top_k)

        docs   = [r[0] for r in results]
        scores = [float(r[1]) for r in results]
        return RetrievedContext(query=query, documents=docs, scores=scores)


# ─────────────────────────────────────────────
# RAG Pipeline (public API)
# ─────────────────────────────────────────────

class RAGPipeline:
    """
    High-level RAG pipeline: ingest documents from any source → retrieve context.

    Supports:
        pipeline.ingest_texts(["text1", "text2"])
        pipeline.ingest_file("document.pdf")
        pipeline.ingest_directory("docs/", glob="**/*.txt")
        pipeline.save_index() / pipeline.load_index()
        context = pipeline.retrieve("What is RAG?")
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        top_k: int = 5,
    ):
        self.processor    = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vector_store = VectorStoreManager(model_name=embedding_model)
        self.top_k        = top_k
        self._is_built    = False

    # ── Ingestion helpers ────────────────────────────────────────────────────

    def ingest_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
    ) -> int:
        """Ingest raw text strings."""
        docs = self.processor.from_texts(texts, metadatas)
        self.vector_store.add(docs)
        self._is_built = True
        return len(docs)

    def ingest_file(self, filepath: str) -> int:
        """Ingest a single .txt or .pdf file."""
        docs = self.processor.from_file(filepath)
        self.vector_store.add(docs)
        self._is_built = True
        return len(docs)

    def ingest_directory(self, dirpath: str, glob: str = "**/*.txt") -> int:
        """Ingest all matching files from a directory."""
        docs = self.processor.from_directory(dirpath, glob=glob)
        self.vector_store.add(docs)
        self._is_built = True
        return len(docs)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> RetrievedContext:
        """Retrieve the top-k most relevant chunks for the query."""
        if not self._is_built:
            raise RuntimeError("Pipeline not built. Call ingest_texts() or ingest_file() first.")
        return self.vector_store.retrieve(query, top_k=self.top_k)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_index(self, path: str = "faiss_index") -> None:
        self.vector_store.save(path)

    def load_index(self, path: str = "faiss_index") -> None:
        self.vector_store.load(path)
        self._is_built = True


# ─────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    DOCS = [
        "Retrieval-Augmented Generation (RAG) combines retrieval with LLM generation.",
        "Hallucination in LLMs refers to generating factually incorrect or unsupported text.",
        "FAISS enables fast approximate nearest-neighbour search over dense embeddings.",
        "Sentence transformers produce dense vector embeddings for semantic similarity.",
        "The faithfulness metric in RAGAS measures whether answers are grounded in context.",
        "Correction loops re-retrieve and regenerate answers when hallucinations are detected.",
        "Self-verification lets an LLM evaluate whether its output is supported by evidence.",
        "Chunking documents into overlapping segments improves retrieval recall in RAG.",
        "A vector store indexes document embeddings for fast semantic search at query time.",
        "The correction engine refines search queries based on unsupported claims.",
    ]

    pipeline = RAGPipeline(chunk_size=400, chunk_overlap=80, top_k=3)
    n = pipeline.ingest_texts(DOCS)
    print(f"\nIngested {n} chunks.\n")

    ctx = pipeline.retrieve("How does hallucination correction work?")
    print(f"Query: {ctx.query}")
    for i, (doc, score) in enumerate(zip(ctx.documents, ctx.scores)):
        print(f"  [{i+1}] ({score:.4f}) {doc.page_content[:80]}...")
