"""
Self-Verifying RAG System — Streamlit Interface (Research-Grade)
================================================================
Run with: streamlit run app.py

Features:
  • Chat interface for asking questions
  • File upload (TXT/PDF) and folder ingestion
  • Color-coded per-sentence verification badges
  • Hallucination score displayed as progress bar
  • Expandable retrieved evidence and correction attempt logs
"""

import streamlit as st
import time

from hallucination_detector import SelfVerifyingRAG, VerificationStatus

# ─────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Self-Verifying RAG — Local LLM",
    page_icon="🧠",
    layout="wide",
)

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────

if "system" not in st.session_state:
    st.session_state.system = None

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Welcome! Load a model and ingest documents in the sidebar, then ask a question."}
    ]

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

STATUS_COLOR = {
    "SUPPORTED":   "🟢",
    "PARTIAL":     "🟡",
    "UNSUPPORTED": "🔴",
}

def score_color(score: float) -> str:
    if score < 0.2:
        return "green"
    if score < 0.5:
        return "orange"
    return "red"

# ─────────────────────────────────────────────
# Sidebar — Configuration & Knowledge Ingestion
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Model Settings")

    model_choice = st.selectbox(
        "Local LLM",
        [
            "HuggingFaceTB/SmolLM-135M-Instruct",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "microsoft/Phi-3-mini-4k-instruct",
        ],
    )
    threshold  = st.slider("Hallucination Threshold", 0.0, 1.0, 0.3, 0.05,
                            help="Scores above this trigger the correction loop.")
    max_retries = st.number_input("Max Correction Retries", 0, 5, 2)
    top_k       = st.number_input("Top-K Retrieval Chunks", 1, 10, 5)

    if st.button("🚀 Load System", type="primary"):
        with st.spinner(f"Loading {model_choice}…"):
            st.session_state.system = SelfVerifyingRAG(
                llm_model=model_choice,
                hallucination_threshold=threshold,
                max_retries=int(max_retries),
                top_k=int(top_k),
            )
        st.success("System ready!")

    st.divider()

    # ── Knowledge ingestion ───────────────────────────────────────────────────

    st.header("📚 Knowledge Base")

    tab_text, tab_file, tab_dir = st.tabs(["📝 Paste Text", "📄 Upload File", "📂 Directory"])

    with tab_text:
        user_text = st.text_area(
            "One document per line:",
            height=150,
            placeholder="Paste knowledge base chunks here…",
        )
        if st.button("Ingest Text"):
            if st.session_state.system is None:
                st.error("Load the system first!")
            else:
                chunks = [t.strip() for t in user_text.split("\n") if len(t.strip()) > 10]
                if chunks:
                    n = st.session_state.system.rag.ingest_texts(chunks)
                    st.success(f"Ingested {n} chunks.")
                else:
                    st.warning("No valid text found.")

    with tab_file:
        uploaded = st.file_uploader("TXT or PDF file", type=["txt", "pdf"])
        if st.button("Ingest File") and uploaded:
            if st.session_state.system is None:
                st.error("Load the system first!")
            else:
                import tempfile, os
                suffix = os.path.splitext(uploaded.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                    f.write(uploaded.read())
                    tmp_path = f.name
                n = st.session_state.system.rag.ingest_file(tmp_path)
                os.unlink(tmp_path)
                st.success(f"Ingested {n} chunks from {uploaded.name}.")

    with tab_dir:
        dir_path = st.text_input("Folder path", placeholder="C:/my_docs")
        dir_glob = st.text_input("File glob", value="**/*.txt")
        if st.button("Ingest Directory"):
            if st.session_state.system is None:
                st.error("Load the system first!")
            elif not dir_path:
                st.warning("Enter a directory path.")
            else:
                n = st.session_state.system.rag.ingest_directory(dir_path, dir_glob)
                st.success(f"Ingested {n} chunks from directory.")


# ─────────────────────────────────────────────
# Main — Chat Interface
# ─────────────────────────────────────────────

st.title("🧠 Self-Verifying Local LLM")
st.markdown(
    "Ask a question about your ingested knowledge. "
    "The system detects and corrects hallucinations at the **claim level**."
)

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Ask a question…"):

    if st.session_state.system is None:
        st.error("Load the system from the sidebar first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        # Guard: knowledge base must be populated
        if not st.session_state.system.rag._is_built:
            st.warning("⚠️ Please ingest documents first (use the sidebar).")
            st.session_state.messages.append(
                {"role": "assistant", "content": "⚠️ Please ingest documents first."}
            )
            st.stop()

        with st.status("Retrieving, generating and verifying…", expanded=True) as status:
            st.write("🔍 Retrieving relevant context…")
            t0 = time.time()
            result = st.session_state.system.ask(prompt)
            elapsed = time.time() - t0
            status.update(label=f"Done in {elapsed:.1f}s", state="complete", expanded=False)

        # ── Final answer ──────────────────────────────────────────────────────
        st.markdown(f"**Answer:**\n\n{result.final_answer}")

        # ── Metrics ───────────────────────────────────────────────────────────
        final_v = result.verification_history[-1]
        score   = final_v.hallucination_score

        col1, col2, col3 = st.columns(3)
        col1.metric("Hallucination Score", f"{score:.2f}",
                    delta=None, delta_color="inverse")
        col2.metric("Correction Attempts", result.attempts - 1)
        col3.metric("Was Corrected?", "Yes ✓" if result.is_corrected else "No")

        # Score bar
        st.markdown(f"**Hallucination Level:** :{score_color(score)}[{'█' * int(score*10)}{'░' * (10-int(score*10))}] {score*100:.0f}%")

        st.divider()

        # ── Per-sentence verification ─────────────────────────────────────────
        with st.expander("🔬 Sentence Verification Breakdown", expanded=True):
            for sv in final_v.sentence_verifications:
                icon  = STATUS_COLOR[sv.status.value]
                badge_color = {"SUPPORTED": "green", "PARTIAL": "orange", "UNSUPPORTED": "red"}[sv.status.value]
                st.markdown(f"{icon} :{badge_color}[**{sv.status.value}**] — {sv.sentence}")
                st.caption(f"Reason: {sv.reason}")

                if sv.claim_verifications:
                    for cv in sv.claim_verifications:
                        cv_icon = STATUS_COLOR[cv.status.value]
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{cv_icon} *Claim:* {cv.claim}")

        # ── Retrieved evidence ────────────────────────────────────────────────
        with st.expander("📄 Retrieved Evidence"):
            ctx = st.session_state.system.rag.retrieve(prompt)
            for i, (doc, sc) in enumerate(zip(ctx.documents, ctx.scores)):
                st.markdown(f"**[{i+1}] Similarity score: {sc:.4f}**")
                st.info(doc.page_content)

        # ── Correction loop log ───────────────────────────────────────────────
        if len(result.verification_history) > 1:
            with st.expander("🔄 Correction Attempts"):
                for i, ver in enumerate(result.verification_history):
                    label = "Initial Answer" if i == 0 else f"Attempt {i}"
                    delta = ""
                    if i > 0:
                        prev_score = result.verification_history[i-1].hallucination_score
                        curr_score = ver.hallucination_score
                        arrow = "⬇️" if curr_score < prev_score else "⬆️"
                        delta = f" {arrow} {prev_score:.2f} → {curr_score:.2f}"
                    st.markdown(f"**{label}** — Score: {ver.hallucination_score:.2f}{delta}")
                    st.markdown(f"> {ver.answer[:200]}{'…' if len(ver.answer) > 200 else ''}")

        # Save to message history
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"{result.final_answer}\n\n*(Hallucination score: {score:.2f} | Attempts: {result.attempts})*",
        })
