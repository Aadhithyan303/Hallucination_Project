"""
Self-Verifying RAG System — Evaluation (Research-Grade)
=======================================================
Improvements over v1:
  • Built-in 25-question domain Q&A dataset
  • Measures hallucination rate, correction success rate, avg hallucination score
  • Saves full results to evaluation_report.json
"""

import os
import json
import statistics
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from hallucination_detector import SelfVerifyingRAG, CorrectedAnswer, VerificationStatus


# ─────────────────────────────────────────────
# Built-in test dataset (25 questions)
# ─────────────────────────────────────────────

DEFAULT_TEST_SET: List[Dict] = [
    {"question": "What is Retrieval-Augmented Generation?",
     "ground_truth": "RAG combines retrieval systems with language model generation."},
    {"question": "What is hallucination in LLMs?",
     "ground_truth": "Hallucination is generating factually incorrect or unsupported text."},
    {"question": "How does self-verification work in RAG?",
     "ground_truth": "The LLM checks its own output for factual consistency with retrieved documents."},
    {"question": "What is FAISS?",
     "ground_truth": "FAISS is a library for fast approximate nearest-neighbour search over dense embeddings."},
    {"question": "What are sentence transformers?",
     "ground_truth": "Sentence transformers produce dense vector embeddings for semantic similarity."},
    {"question": "What is the faithfulness metric in RAGAS?",
     "ground_truth": "Faithfulness measures whether the answer is grounded in the retrieved context."},
    {"question": "What is answer relevancy in RAGAS?",
     "ground_truth": "Answer relevancy measures how relevant the generated answer is to the question."},
    {"question": "What is a correction loop?",
     "ground_truth": "A correction loop re-retrieves documents and regenerates the answer when hallucinations are detected."},
    {"question": "What is a hallucination score?",
     "ground_truth": "The hallucination score is the average of individual claim scores, where 0 means fully grounded."},
    {"question": "What is chunking in RAG?",
     "ground_truth": "Chunking splits documents into overlapping segments to improve retrieval recall."},
    {"question": "What is a vector store?",
     "ground_truth": "A vector store indexes document embeddings for fast semantic search at query time."},
    {"question": "How does claim-level verification improve hallucination detection?",
     "ground_truth": "Claim-level verification breaks sentences into atomic facts and verifies each separately."},
    {"question": "What does SUPPORTED mean in verification?",
     "ground_truth": "SUPPORTED means the claim is fully backed by the retrieved context."},
    {"question": "What does UNSUPPORTED mean in verification?",
     "ground_truth": "UNSUPPORTED means the claim contradicts or is absent from the retrieved context."},
    {"question": "What does PARTIAL mean in verification?",
     "ground_truth": "PARTIAL means the claim is only partially backed by the retrieved context."},
    {"question": "What is query refinement?",
     "ground_truth": "Query refinement rewrites the search query to target unsupported claims specifically."},
    {"question": "Why does RAG reduce hallucination?",
     "ground_truth": "RAG grounds generation in retrieved documents, reducing the chance of fabricating facts."},
    {"question": "What embedding model is used?",
     "ground_truth": "The system uses sentence-transformers/all-MiniLM-L6-v2 for embeddings."},
    {"question": "What is the default chunk size?",
     "ground_truth": "The default chunk size is 800 characters with 150-character overlap."},
    {"question": "What happens when a hallucination is detected?",
     "ground_truth": "The correction engine extracts failed claims, refines the query, re-retrieves, and regenerates."},
    {"question": "What is the correction success rate?",
     "ground_truth": "The correction success rate is the fraction of corrections that reduced the hallucination score."},
    {"question": "What document formats are supported?",
     "ground_truth": "The system supports TXT and PDF files as well as directory ingestion."},
    {"question": "What is the hallucination-free rate?",
     "ground_truth": "The hallucination-free rate is the fraction of questions answered with a score of 0."},
    {"question": "What local models are supported?",
     "ground_truth": "Any HuggingFace text-generation model can be used, such as SmolLM or Qwen."},
    {"question": "How is the final hallucination score computed?",
     "ground_truth": "The score is the mean of all individual claim scores across all sentences in the answer."},
]

SAMPLE_KNOWLEDGE_BASE: List[str] = [
    "Retrieval-Augmented Generation (RAG) combines a retrieval system with an LLM to answer questions using external knowledge.",
    "Hallucination in large language models refers to generating text that sounds plausible but is factually incorrect or unsupported by evidence.",
    "Self-verification is a technique where the LLM evaluates its own output for factual consistency with retrieved documents.",
    "FAISS (Facebook AI Similarity Search) enables fast approximate nearest-neighbor search over dense vector embeddings.",
    "The RAGAS framework provides metrics like faithfulness, answer relevancy, and context recall for evaluating RAG pipelines.",
    "Correction loops in self-verifying RAG re-retrieve documents with a refined query and regenerate the answer when hallucinations are detected.",
    "Sentence transformers convert text into dense embeddings that capture semantic similarity, useful for retrieval.",
    "The hallucination score is computed as the mean of all individual claim scores (SUPPORTED=0, PARTIAL=0.5, UNSUPPORTED=1.0).",
    "Chunking splits documents into overlapping text segments to ensure retrieval captures complete context.",
    "A vector store indexes document embeddings for fast semantic search at query time.",
    "Claim-level verification breaks complex sentences into atomic factual claims and verifies each claim separately.",
    "SUPPORTED means the claim is fully and directly backed by the retrieved context.",
    "UNSUPPORTED means the claim contradicts or is absent from the retrieved context.",
    "PARTIAL means the claim is only partially backed by the retrieved context.",
    "Query refinement rewrites the retrieval search query to specifically target unsupported claims.",
    "RAG reduces hallucination by grounding LLM generation in retrieved documents rather than relying on parametric memory.",
    "The default embedding model is sentence-transformers/all-MiniLM-L6-v2.",
    "The default chunk size in the research-grade pipeline is 800 characters with 150-character overlap.",
    "When a hallucination is detected, the correction engine extracts failed claims, refines the query, re-retrieves, and regenerates.",
    "The correction success rate measures the fraction of correction attempts that reduced the hallucination score.",
    "The system supports ingestion of TXT files, PDF documents, and full directories of documents.",
    "The hallucination-free rate measures the fraction of answers with a hallucination score of 0.",
    "Any HuggingFace text-generation model can be used with this system, such as SmolLM-135M-Instruct or Qwen2.5.",
    "The final hallucination score is the mean of individual claim scores across all sentences in the generated answer.",
]


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class SingleEvalResult:
    question: str
    ground_truth: str
    generated_answer: str
    hallucination_score: float
    correction_attempted: bool
    correction_succeeded: bool
    attempts: int


@dataclass
class EvaluationReport:
    results: List[SingleEvalResult]

    @property
    def avg_hallucination_score(self) -> float:
        return statistics.mean(r.hallucination_score for r in self.results)

    @property
    def correction_rate(self) -> float:
        attempted = [r for r in self.results if r.correction_attempted]
        if not attempted:
            return 0.0
        return sum(1 for r in attempted if r.correction_succeeded) / len(attempted)

    @property
    def hallucination_free_rate(self) -> float:
        return sum(1 for r in self.results if r.hallucination_score == 0.0) / len(self.results)

    @property
    def hallucination_rate(self) -> float:
        """Fraction of answers that needed correction."""
        return sum(1 for r in self.results if r.correction_attempted) / len(self.results)

    def summary(self) -> Dict:
        return {
            "total_questions":         len(self.results),
            "avg_hallucination_score": round(self.avg_hallucination_score, 4),
            "hallucination_rate":      round(self.hallucination_rate, 4),
            "hallucination_free_rate": round(self.hallucination_free_rate, 4),
            "correction_rate":         round(self.correction_rate, 4),
        }

    def print_report(self) -> None:
        s = self.summary()
        print("\n" + "="*55)
        print("  EVALUATION REPORT")
        print("="*55)
        print(f"  Total Questions        : {s['total_questions']}")
        print(f"  Avg Hallucination Score: {s['avg_hallucination_score']:.4f}  (0=best, 1=worst)")
        print(f"  Hallucination Rate     : {s['hallucination_rate']*100:.1f}%  (answers needing correction)")
        print(f"  Hallucination-Free Rate: {s['hallucination_free_rate']*100:.1f}%")
        print(f"  Correction Success Rate: {s['correction_rate']*100:.1f}%")
        print("="*55)
        print("\nPer-question results:")
        for r in self.results:
            tag = "✓" if r.hallucination_score == 0 else f"✗ score={r.hallucination_score:.2f}"
            print(f"  {tag} | attempts={r.attempts} | Q: {r.question[:50]}")


# ─────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────

class Evaluator:
    """Runs the self-verifying RAG system over a test dataset and collects metrics."""

    def __init__(self, system: SelfVerifyingRAG):
        self.system = system

    def evaluate(self, test_data: Optional[List[Dict]] = None) -> EvaluationReport:
        if test_data is None:
            test_data = DEFAULT_TEST_SET

        results: List[SingleEvalResult] = []
        for i, item in enumerate(test_data):
            q  = item["question"]
            gt = item.get("ground_truth", "")
            print(f"\n[Eval] [{i+1}/{len(test_data)}] {q}")

            result: CorrectedAnswer = self.system.ask(q)
            final_v = result.verification_history[-1]
            first_v = result.verification_history[0]

            correction_attempted = len(result.verification_history) > 1
            correction_succeeded = (
                correction_attempted
                and final_v.hallucination_score < first_v.hallucination_score
            )

            results.append(SingleEvalResult(
                question=q,
                ground_truth=gt,
                generated_answer=result.final_answer,
                hallucination_score=final_v.hallucination_score,
                correction_attempted=correction_attempted,
                correction_succeeded=correction_succeeded,
                attempts=result.attempts,
            ))

        return EvaluationReport(results=results)

    def save_report(self, report: EvaluationReport, path: str = "evaluation_report.json") -> None:
        data = {
            "summary": report.summary(),
            "per_question": [
                {
                    "question":              r.question,
                    "ground_truth":          r.ground_truth,
                    "generated_answer":      r.generated_answer,
                    "hallucination_score":   r.hallucination_score,
                    "correction_attempted":  r.correction_attempted,
                    "correction_succeeded":  r.correction_succeeded,
                    "attempts":              r.attempts,
                }
                for r in report.results
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"\n[Eval] Report saved to '{path}'")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    system = SelfVerifyingRAG(
        llm_model="HuggingFaceTB/SmolLM-135M-Instruct",
        hallucination_threshold=0.3,
        max_retries=2,
    )
    system.ingest(SAMPLE_KNOWLEDGE_BASE)

    evaluator = Evaluator(system)
    # Run on first 5 questions for a quick smoke-test; remove the slice for full evaluation
    report = evaluator.evaluate(DEFAULT_TEST_SET[:5])
    report.print_report()
    evaluator.save_report(report, "evaluation_report.json")
