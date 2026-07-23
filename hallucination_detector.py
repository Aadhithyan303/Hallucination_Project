"""
Self-Verifying RAG System — Hallucination Detector (Research-Grade)
====================================================================
Improvements over v1:
  • Claim extraction: sentences split into atomic factual claims before verification
  • Claim-level verification with weighted scoring (SUPPORTED=0, PARTIAL=0.5, UNSUPPORTED=1)
  • Final hallucination_score = mean of all claim scores
  • Simple keyword parser — robust against unreliable local LLM JSON output
  • Clean plain-text prompts with no special instruction tokens
"""

import re
from typing import List, Optional
from dataclasses import dataclass, field
from enum import Enum

import torch
from transformers import pipeline

from rag_pipeline import RAGPipeline, RetrievedContext


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

class VerificationStatus(str, Enum):
    SUPPORTED   = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    PARTIAL     = "PARTIAL"


@dataclass
class ClaimVerification:
    """Verification result for one atomic claim extracted from a sentence."""
    claim: str
    status: VerificationStatus
    confidence: float = 0.8


@dataclass
class SentenceVerification:
    """Aggregated verification result for one sentence (may contain many claims)."""
    sentence: str
    claim_verifications: List[ClaimVerification]
    status: VerificationStatus
    reason: str
    confidence: float

    @classmethod
    def from_claims(cls, sentence: str, claims: List[ClaimVerification]) -> "SentenceVerification":
        """Aggregate claim verdicts into a single sentence verdict."""
        if not claims:
            return cls(sentence=sentence, claim_verifications=[],
                       status=VerificationStatus.PARTIAL,
                       reason="No claims extracted.", confidence=0.5)

        weights  = {VerificationStatus.SUPPORTED: 0.0,
                    VerificationStatus.PARTIAL:   0.5,
                    VerificationStatus.UNSUPPORTED: 1.0}
        avg_score = sum(weights[c.status] for c in claims) / len(claims)

        if avg_score == 0.0:
            status = VerificationStatus.SUPPORTED
        elif avg_score >= 0.5:
            status = VerificationStatus.UNSUPPORTED
        else:
            status = VerificationStatus.PARTIAL

        reason = f"{sum(1 for c in claims if c.status == VerificationStatus.SUPPORTED)}/{len(claims)} claims supported."
        return cls(sentence=sentence, claim_verifications=claims,
                   status=status, reason=reason, confidence=1 - avg_score)


@dataclass
class VerificationResult:
    """Full verification result for a generated answer."""
    answer: str
    context: str
    sentence_verifications: List[SentenceVerification]
    overall_status: VerificationStatus
    hallucination_score: float   # 0.0 = fully grounded, 1.0 = fully hallucinated
    correction_needed: bool

    def summary(self) -> str:
        lines = [
            f"Overall Status  : {self.overall_status.value}",
            f"Hallucination % : {self.hallucination_score * 100:.1f}%",
            f"Needs Correction: {self.correction_needed}",
            "",
            "Sentence-level breakdown:",
        ]
        for sv in self.sentence_verifications:
            icon = {"SUPPORTED": "✓", "UNSUPPORTED": "✗", "PARTIAL": "~"}[sv.status.value]
            lines.append(f"  {icon} [{sv.status.value}] {sv.sentence[:70]}")
            lines.append(f"      {sv.reason}")
        return "\n".join(lines)


@dataclass
class CorrectedAnswer:
    """Output of the correction loop."""
    original_answer: str
    final_answer: str
    attempts: int
    is_corrected: bool
    verification_history: List[VerificationResult]


# ─────────────────────────────────────────────
# Local LLM Client
# ─────────────────────────────────────────────

class LocalLLMClient:
    """
    Local LLM generation via HuggingFace Transformers.
    Uses plain-text prompts — no special tokens — for universal model compatibility.
    """

    def __init__(self, model_id: str = "HuggingFaceTB/SmolLM-135M-Instruct"):
        print(f"[LLM] Loading model: {model_id}...")
        self.model_id = model_id
        self.device   = 0 if torch.cuda.is_available() else -1

        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            device=self.device,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        print(f"[LLM] Ready on {'GPU' if self.device == 0 else 'CPU'}")

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Run text generation and return only the new tokens."""
        results = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            return_full_text=False,
        )
        return results[0]["generated_text"].strip()


# ─────────────────────────────────────────────
# Answer Generator
# ─────────────────────────────────────────────

_RETRIEVAL_MARKERS = ["[Source", "<|", "|>", "[/Source", "Context:", "Question:"]

def _clean_answer(text: str) -> str:
    """Strip any echoed retrieval markers or prompt artifacts from generated text."""
    for marker in _RETRIEVAL_MARKERS:
        if marker in text:
            text = text.split(marker)[0]
    return text.strip()


class AnswerGenerator:
    """
    Generates a grounded answer using a strict plain-text prompt.
    Refuses to answer if context does not contain relevant information.
    """

    def __init__(self, llm: LocalLLMClient):
        self.llm = llm

    def generate(self, query: str, context: str) -> str:
        prompt = (
            "Answer the question using ONLY the information in the context below.\n"
            "If the context does not contain enough information, reply exactly:\n"
            "  I don't know based on the provided context.\n"
            "Do NOT repeat the context. Do NOT add source markers.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "Answer:"
        )
        raw = self.llm.generate(prompt, max_new_tokens=256)
        return _clean_answer(raw)


# ─────────────────────────────────────────────
# Hallucination Detector (Research-Grade)
# ─────────────────────────────────────────────

class HallucinationDetector:
    """
    Research-grade hallucination detector with claim-level verification.

    Pipeline per sentence:
      sentence → extract atomic claims → verify each claim → aggregate scores
    """

    def __init__(self, llm: LocalLLMClient, hallucination_threshold: float = 0.3):
        self.llm = llm
        self.hallucination_threshold = hallucination_threshold

    # ── Sentence splitting ────────────────────────────────────────────────────

    def _split_sentences(self, text: str) -> List[str]:
        """Split answer into individual sentences."""
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in parts if len(s.strip()) > 10]

    # ── Claim extraction ──────────────────────────────────────────────────────

    def _extract_claims(self, sentence: str) -> List[str]:
        """
        Split a sentence into atomic, independently verifiable factual claims.
        Falls back to the whole sentence if the model returns garbage.
        """
        prompt = (
            "Break the following sentence into a list of simple, atomic factual claims.\n"
            "Write one claim per line. Do not add explanations.\n\n"
            f"Sentence: {sentence}\n\n"
            "Claims:"
        )
        raw = self.llm.generate(prompt, max_new_tokens=128)

        # Parse line-by-line; filter blank / too-short lines
        claims = [
            line.lstrip("•-–*0123456789. ").strip()
            for line in raw.splitlines()
            if len(line.strip()) > 8
        ]
        # Cap at 5 claims per sentence to avoid runaway generation
        return claims[:5] if claims else [sentence]

    # ── Claim verification ────────────────────────────────────────────────────

    def _verify_claim(self, claim: str, context: str) -> ClaimVerification:
        """
        Verify one atomic claim against the retrieved context.
        Uses a one-word prompt, parsed by keyword matching — no JSON needed.
        """
        prompt = (
            "Is the following statement supported by the context?\n\n"
            f"Context:\n{context}\n\n"
            f"Statement: {claim}\n\n"
            "Reply with ONE word — SUPPORTED, UNSUPPORTED, or PARTIAL:"
        )
        raw = self.llm.generate(prompt, max_new_tokens=8).upper().strip()

        # Keyword match order matters: check UNSUPPORTED before SUPPORTED
        if "UNSUPPORTED" in raw:
            status = VerificationStatus.UNSUPPORTED
        elif "SUPPORTED" in raw:
            status = VerificationStatus.SUPPORTED
        else:
            status = VerificationStatus.PARTIAL

        return ClaimVerification(claim=claim, status=status)

    # ── Sentence verification ─────────────────────────────────────────────────

    def _verify_sentence(self, sentence: str, context: str) -> SentenceVerification:
        """Extract claims from a sentence and verify each one."""
        claims_text = self._extract_claims(sentence)
        claim_verifs = [self._verify_claim(c, context) for c in claims_text]
        return SentenceVerification.from_claims(sentence, claim_verifs)

    # ── Full answer verification ──────────────────────────────────────────────

    def verify(self, answer: str, context: str) -> VerificationResult:
        """
        Verify all sentences in the answer at the claim level.
        Returns a VerificationResult with weighted hallucination_score.
        """
        sentences = self._split_sentences(answer)
        sent_verifs: List[SentenceVerification] = []

        print(f"[Detector] Verifying {len(sentences)} sentence(s) at claim level...")
        for sent in sentences:
            sv = self._verify_sentence(sent, context)
            sent_verifs.append(sv)
            print(f"  -> {sv.status.value} ({sv.reason}): {sent[:55]}...")

        # Aggregate hallucination score across all claims in all sentences
        weights = {VerificationStatus.SUPPORTED: 0.0,
                   VerificationStatus.PARTIAL:   0.5,
                   VerificationStatus.UNSUPPORTED: 1.0}

        all_claims = [c for sv in sent_verifs for c in sv.claim_verifications]
        if all_claims:
            hallucination_score = sum(weights[c.status] for c in all_claims) / len(all_claims)
        else:
            hallucination_score = 0.0

        n_unsupported = sum(1 for sv in sent_verifs if sv.status == VerificationStatus.UNSUPPORTED)
        n_partial     = sum(1 for sv in sent_verifs if sv.status == VerificationStatus.PARTIAL)

        if n_unsupported == 0 and n_partial == 0:
            overall = VerificationStatus.SUPPORTED
        elif n_unsupported > 0:
            overall = VerificationStatus.UNSUPPORTED
        else:
            overall = VerificationStatus.PARTIAL

        return VerificationResult(
            answer=answer,
            context=context,
            sentence_verifications=sent_verifs,
            overall_status=overall,
            hallucination_score=hallucination_score,
            correction_needed=hallucination_score >= self.hallucination_threshold,
        )


# ─────────────────────────────────────────────
# Import correction engine
# ─────────────────────────────────────────────

from correction_engine import CorrectionEngine


# ─────────────────────────────────────────────
# Self-Verifying RAG — Orchestrator
# ─────────────────────────────────────────────

class SelfVerifyingRAG:
    """
    Complete self-verifying RAG system with a local LLM.

    Usage:
        system = SelfVerifyingRAG()
        system.ingest(["doc1 text", "doc2 text"])
        result = system.ask("What is hallucination?")
        print(result.final_answer)
    """

    def __init__(
        self,
        llm_model: str = "HuggingFaceTB/SmolLM-135M-Instruct",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        hallucination_threshold: float = 0.3,
        max_retries: int = 2,
        top_k: int = 5,
    ):
        self.llm       = LocalLLMClient(model_id=llm_model)
        self.rag       = RAGPipeline(embedding_model=embedding_model, top_k=top_k)
        self.generator = AnswerGenerator(self.llm)
        self.detector  = HallucinationDetector(self.llm, hallucination_threshold)
        self.corrector = CorrectionEngine(self.rag, self.generator, self.detector, max_retries)

    # ── Ingestion helpers ─────────────────────────────────────────────────────

    def ingest(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> None:
        n = self.rag.ingest_texts(texts, metadatas)
        print(f"[System] Ingested {n} chunks.")

    def ingest_file(self, filepath: str) -> None:
        n = self.rag.ingest_file(filepath)
        print(f"[System] Ingested {n} chunks from file.")

    def ingest_directory(self, dirpath: str, glob: str = "**/*.txt") -> None:
        n = self.rag.ingest_directory(dirpath, glob)
        print(f"[System] Ingested {n} chunks from directory.")

    # ── Query ─────────────────────────────────────────────────────────────────

    def ask(self, query: str) -> CorrectedAnswer:
        """Full pipeline: retrieve → generate → verify → correct if needed."""
        print(f"\n{'='*55}\nQuery: {query}\n{'='*55}")

        ctx         = self.rag.retrieve(query)
        context_str = ctx.to_context_string()
        print(f"[System] Retrieved {len(ctx.documents)} chunks.")

        initial_answer = self.generator.generate(query, context_str)
        print(f"\n[System] Initial answer:\n  {initial_answer}\n")

        verification = self.detector.verify(initial_answer, context_str)
        print(f"\n[System] Verification:\n{verification.summary()}\n")

        if verification.correction_needed:
            print("[System] Correction triggered...")
            return self.corrector.correct(query, initial_answer, verification)

        return CorrectedAnswer(
            original_answer=initial_answer,
            final_answer=initial_answer,
            attempts=1,
            is_corrected=False,
            verification_history=[verification],
        )


# ─────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    DOCS = [
        "Retrieval-Augmented Generation (RAG) combines retrieval with LLM generation.",
        "Hallucination in LLMs refers to generating text that is factually incorrect.",
        "Self-verification lets the LLM check its own output against retrieved evidence.",
        "FAISS enables fast approximate nearest-neighbour search over dense embeddings.",
        "Correction loops re-retrieve and regenerate answers when hallucinations are found.",
        "Sentence transformers produce dense vector embeddings for semantic search.",
        "The hallucination score is the average of individual claim scores (0=grounded, 1=hallucinated).",
    ]

    system = SelfVerifyingRAG(hallucination_threshold=0.3, max_retries=1)
    system.ingest(DOCS)
    result = system.ask("How does hallucination correction work?")

    print("\n" + "="*55)
    print("FINAL RESULT")
    print("="*55)
    print(f"Answer : {result.final_answer}")
    print(f"Score  : {result.verification_history[-1].hallucination_score:.2f}")
    print(f"Corrected: {result.is_corrected}")
