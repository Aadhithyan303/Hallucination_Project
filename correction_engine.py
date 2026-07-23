"""
Self-Verifying RAG System — Correction Engine (Research-Grade)
==============================================================
Improvements over v1:
  • Extracts specific unsupported claims before building the refined query
  • Builds a focused, targeted re-retrieval query from failed claims
  • Logs before/after hallucination scores for every attempt
  • Strips retrieval markers from regenerated answers
"""

from typing import List
from hallucination_detector import (
    VerificationResult,
    VerificationStatus,
    CorrectedAnswer,
    _clean_answer,
)


class CorrectionEngine:
    """
    Correction loop:
      detect hallucination → extract failed claims → refine query
      → re-retrieve → regenerate → re-verify   (repeat up to max_retries)
    """

    def __init__(self, rag_pipeline, generator, detector, max_retries: int = 2):
        self.rag          = rag_pipeline
        self.generator    = generator
        self.detector     = detector
        self.max_retries  = max_retries

    # ── Query refinement ──────────────────────────────────────────────────────

    def _refine_query(self, original_query: str, failed_claims: List[str]) -> str:
        """
        Ask the local LLM for a focused retrieval query that targets the
        specific unsupported claims rather than the broad original question.
        """
        claims_text = "\n".join(f"- {c}" for c in failed_claims[:4])
        prompt = (
            "You are helping improve a search query to find missing evidence.\n\n"
            f"Original question:\n{original_query}\n\n"
            f"These factual claims were NOT supported by the previous search results:\n{claims_text}\n\n"
            "Write ONE concise search query (no more than 15 words) that would retrieve evidence "
            "for those unsupported claims:\n"
        )
        raw = self.detector.llm.generate(prompt, max_new_tokens=32)
        # Strip common model artefacts (quotes, "Query:", etc.)
        for prefix in ["Query:", "Search:", "Question:", '"', "'"]:
            raw = raw.replace(prefix, "")
        return raw.strip()

    # ── Collect failed claims ─────────────────────────────────────────────────

    @staticmethod
    def _collect_failed_claims(verification: VerificationResult) -> List[str]:
        """Return all unsupported/partial claim texts from a VerificationResult."""
        failed: List[str] = []
        for sv in verification.sentence_verifications:
            for cv in sv.claim_verifications:
                if cv.status in (VerificationStatus.UNSUPPORTED, VerificationStatus.PARTIAL):
                    failed.append(cv.claim)
        # Fall back to whole unsupported sentences if no atomic claims available
        if not failed:
            for sv in verification.sentence_verifications:
                if sv.status in (VerificationStatus.UNSUPPORTED, VerificationStatus.PARTIAL):
                    failed.append(sv.sentence)
        return failed

    # ── Correction loop ───────────────────────────────────────────────────────

    def correct(
        self,
        query: str,
        initial_answer: str,
        initial_verification: VerificationResult,
    ) -> CorrectedAnswer:
        """
        Run the correction loop until the answer is verified or max_retries exhausted.
        """
        history          = [initial_verification]
        current_answer   = initial_answer
        current_verif    = initial_verification

        for attempt in range(1, self.max_retries + 1):
            if not current_verif.correction_needed:
                print(f"[Correction] Verified after attempt {attempt}. Stopping.")
                break

            score_before = current_verif.hallucination_score
            print(f"\n[Correction] Attempt {attempt}/{self.max_retries} "
                  f"(score={score_before:.2f})")

            # 1. Collect failed claims
            failed_claims = self._collect_failed_claims(current_verif)
            print(f"[Correction] {len(failed_claims)} unsupported/partial claim(s).")

            # 2. Refine query
            refined_query = self._refine_query(query, failed_claims)
            print(f"[Correction] Refined query: '{refined_query}'")

            # 3. Re-retrieve with focused query
            ctx         = self.rag.retrieve(refined_query)
            context_str = ctx.to_context_string()

            # 4. Regenerate with correction instruction
            correction_prompt = (
                "The previous answer contained unsupported claims.\n"
                "Using ONLY the context below, write a corrected, grounded answer.\n"
                "Do not repeat the context. Do not add source markers.\n\n"
                f"Context:\n{context_str}\n\n"
                f"Question:\n{query}\n\n"
                "Corrected answer:"
            )
            new_answer     = self.detector.llm.generate(correction_prompt, max_new_tokens=256)
            current_answer = _clean_answer(new_answer)

            # 5. Re-verify
            current_verif = self.detector.verify(current_answer, context_str)
            history.append(current_verif)

            score_after = current_verif.hallucination_score
            print(f"[Correction] Score: {score_before:.2f} -> {score_after:.2f}")

        is_corrected = (
            len(history) > 1
            and history[-1].hallucination_score < history[0].hallucination_score
        )

        return CorrectedAnswer(
            original_answer=initial_answer,
            final_answer=current_answer,
            attempts=len(history),
            is_corrected=is_corrected,
            verification_history=history,
        )
