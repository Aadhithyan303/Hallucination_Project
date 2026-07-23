# Self-Verifying Retrieval-Augmented LLMs
### Hallucination Detection and Correction System

---

## Project Overview

This project implements a **Self-Verifying RAG system** that:

1. **Retrieves** relevant documents for a user query (RAG pipeline)
2. **Generates** an initial answer grounded in retrieved context
3. **Detects** hallucinations by verifying each sentence against evidence
4. **Corrects** the answer automatically if hallucinations are found

---

## Architecture

```
User Query
    │
    ▼
RAG Retrieval  ←─── FAISS Vector Store (HuggingFace Embeddings)
    │
    ▼
LLM Generation ─── GPT-3.5 / GPT-4
    │
    ▼
Self-Verification ─── Sentence-level NLI-style checking
    │
    ├── SUPPORTED ──► Final Answer
    │
    └── UNSUPPORTED ─► Correction Loop
                           │
                           ▼
                       Refined Query → Re-retrieve → Regenerate
```

---

## File Structure

```
project/
├── rag_pipeline.py          # Phase 1: RAG pipeline (load, embed, retrieve)
├── hallucination_detector.py # Phase 2 & 3: Detection + correction
├── evaluation.py            # Phase 4: Metrics & RAGAS evaluation
├── requirements.txt         # All dependencies
└── README.md
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key
```bash
export OPENAI_API_KEY="sk-your-key-here"
```
Or create a `.env` file:
```
OPENAI_API_KEY=sk-your-key-here
```

---

## Quick Start

### Run RAG pipeline only (no API key needed)
```bash
python rag_pipeline.py
```

### Run the full self-verifying system
```python
from hallucination_detector import SelfVerifyingRAG

system = SelfVerifyingRAG(
    api_key="sk-your-key",
    hallucination_threshold=0.3,   # flag if >30% sentences unsupported
    max_retries=2                  # max correction attempts
)

# Ingest your documents
system.ingest([
    "RAG combines retrieval with language generation.",
    "Hallucination is generating unsupported facts.",
    # ... add your documents here
])

# Ask a question
result = system.ask("What is hallucination in LLMs?")
print(result.final_answer)
print(f"Hallucination score: {result.verification_history[-1].hallucination_score:.2f}")
```

### Run evaluation
```bash
python evaluation.py
```

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `embedding_model` | `all-MiniLM-L6-v2` | HuggingFace model for embeddings |
| `llm_model` | `gpt-3.5-turbo` | OpenAI model for generation |
| `hallucination_threshold` | `0.3` | Score above which correction triggers |
| `max_retries` | `2` | Max correction loop iterations |
| `top_k` | `4` | Number of chunks retrieved per query |
| `chunk_size` | `500` | Characters per document chunk |
| `chunk_overlap` | `100` | Overlap between consecutive chunks |

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| Hallucination Score | 0=fully grounded, 1=fully hallucinated |
| Hallucination-Free Rate | % of answers with score = 0 |
| Correction Success Rate | % of corrections that reduced hallucination |
| RAGAS Faithfulness | Answer entailed by context (0–1) |
| RAGAS Answer Relevancy | Answer relevant to question (0–1) |

---

## Extending the System

- **Use a local model**: Replace `LLMClient` to use Ollama or LlamaCpp
- **Use ChromaDB**: Swap `VectorStoreManager` to use `langchain_community.vectorstores.Chroma`
- **Add NLI model**: Replace the LLM-based verifier with a cross-encoder (e.g. `cross-encoder/nli-deberta-v3-base`)
- **Add streaming**: Wrap `SelfVerifyingRAG.ask()` with async streaming

---

## References

- Lewis et al. (2020) — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
- Ji et al. (2023) — *Survey of Hallucination in Natural Language Generation*
- Es et al. (2023) — *RAGAS: Automated Evaluation of Retrieval Augmented Generation*
- Manakul et al. (2023) — *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection*
