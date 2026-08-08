"""Recruiter-facing Streamlit demo for the Arabic–English RAG Copilot."""

from __future__ import annotations

import streamlit as st


GITHUB_URL = (
    "https://github.com/Rinosa123/"
    "Bilingual-Enterprise-RAG-Copilot"
)

COLAB_URL = (
    "https://colab.research.google.com/github/"
    "Rinosa123/Bilingual-Enterprise-RAG-Copilot/"
    "blob/main/notebooks/05_end_to_end_rag.ipynb"
)


CASES = {
    "How many annual leave days do full-time employees receive?": {
        "answer": (
            "Full-time employees receive 24 working days of annual leave "
            "after completing one year of service [HR-EN-001-CH-003]."
        ),
        "language": "English",
        "retrieved": (
            "HR-EN-001-CH-003",
            "HR-EN-001-CH-002",
            "HR-EN-001-CH-004",
            "HR-AR-001-CH-005",
            "HR-AR-001-CH-004",
        ),
        "evidence": (
            "HR-EN-001-CH-003",
            "HR-AR-001-CH-005",
            "HR-AR-001-CH-004",
        ),
        "citations": ("HR-EN-001-CH-003",),
        "refused": False,
        "latency": {
            "Dense retrieval": 18.32,
            "BGE reranking": 29.40,
            "Qwen generation": 3041.21,
            "Total pipeline": 3090.40,
        },
    },
    "ما الحد الأقصى لتكلفة الفندق لليلة الواحدة؟": {
        "answer": (
            "الحد الأقصى لتكلفة الفندق لليلة الواحدة هو 450 ريالاً "
            "سعودياً، ما لم تتم الموافقة مسبقاً على مبلغ أعلى "
            "[HR-AR-001-CH-003]."
        ),
        "language": "Arabic",
        "retrieved": (
            "HR-AR-001-CH-003",
            "HR-AR-001-CH-002",
            "HR-AR-001-CH-005",
            "HR-AR-001-CH-004",
            "HR-AR-001-CH-001",
        ),
        "evidence": (
            "HR-AR-001-CH-003",
            "HR-AR-001-CH-002",
            "HR-AR-001-CH-004",
        ),
        "citations": ("HR-AR-001-CH-003",),
        "refused": False,
        "latency": {
            "Dense retrieval": 13.59,
            "BGE reranking": 28.94,
            "Qwen generation": 5326.55,
            "Total pipeline": 5369.81,
        },
    },
    "كم عدد أيام الإجازة السنوية للموظف؟": {
        "answer": (
            "يُمنح الموظف 24 يومًا عملًا من الإجازة السنوية بعد إكمال "
            "سنة خدمة [HR-EN-001-CH-003]."
        ),
        "language": "Arabic",
        "retrieved": (
            "HR-AR-001-CH-004",
            "HR-AR-001-CH-003",
            "HR-AR-001-CH-005",
            "HR-EN-001-CH-003",
            "HR-AR-001-CH-002",
        ),
        "evidence": (
            "HR-EN-001-CH-003",
            "HR-AR-001-CH-004",
            "HR-AR-001-CH-005",
        ),
        "citations": ("HR-EN-001-CH-003",),
        "refused": False,
        "latency": {
            "Dense retrieval": 14.25,
            "BGE reranking": 28.44,
            "Qwen generation": 3712.39,
            "Total pipeline": 3755.72,
        },
    },
    "What is the company's maternity leave policy?": {
        "answer": (
            "I could not find sufficient evidence in the provided "
            "documents to answer this question."
        ),
        "language": "English",
        "retrieved": (
            "HR-EN-001-CH-001",
            "HR-EN-001-CH-003",
            "HR-AR-001-CH-001",
            "HR-EN-001-CH-002",
            "HR-EN-001-CH-005",
        ),
        "evidence": (
            "HR-EN-001-CH-003",
            "HR-AR-001-CH-001",
            "HR-EN-001-CH-001",
        ),
        "citations": (),
        "refused": True,
        "latency": {
            "Dense retrieval": 20.11,
            "BGE reranking": 33.46,
            "Qwen generation": 2437.36,
            "Total pipeline": 2491.97,
        },
    },
}


def load_selected_example() -> None:
    """Copy the selected verified question into the question field."""
    st.session_state.question = st.session_state.example


st.set_page_config(
    page_title="Arabic–English RAG Copilot",
    page_icon="🌐",
    layout="wide",
)

st.title("🌐 Arabic–English Enterprise RAG Copilot")
st.write(
    "Multilingual retrieval, cross-language reranking, grounded generation, "
    "citation validation and safe refusal handling."
)

st.info(
    "This hosted interface displays verified Tesla T4 evaluation snapshots. "
    "Use the Colab notebook for live GPU inference."
)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Unit tests", "29 passing")
metric_2.metric("End-to-end checks", "5/5")
metric_3.metric("Languages", "Arabic + English")
metric_4.metric("Citation policy", "Validated")

st.sidebar.header("Pipeline")
st.sidebar.markdown(
    """
1. Multilingual E5 retrieval
2. BGE cross-encoder reranking
3. Qwen grounded generation
4. Citation validation
5. Safe refusal handling
"""
)

st.sidebar.markdown(f"[View source code]({GITHUB_URL})")
st.sidebar.markdown(f"[Run live GPU notebook]({COLAB_URL})")

questions = list(CASES)

if "example" not in st.session_state:
    st.session_state.example = questions[0]

if "question" not in st.session_state:
    st.session_state.question = questions[0]

st.subheader("Try a verified evaluation case")

st.selectbox(
    "Example question",
    questions,
    key="example",
    on_change=load_selected_example,
)

st.text_area(
    "Question",
    key="question",
    height=90,
)

if st.button("Run RAG demo", type="primary"):
    question = st.session_state.question.strip()
    case = CASES.get(question)

    if case is None:
        st.warning(
            "This public app currently uses verified evaluation snapshots. "
            "Select one of the example questions above, or open the Colab "
            "notebook for live inference."
        )
    else:
        st.subheader("Answer")

        if case["refused"]:
            st.warning(case["answer"])
        else:
            st.success(case["answer"])

        result_1, result_2, result_3, result_4 = st.columns(4)
        result_1.metric("Answer language", case["language"])
        result_2.metric(
            "Refused",
            "Yes" if case["refused"] else "No",
        )
        result_3.metric("Citation validation", "Passed")
        result_4.metric("Safety blocked", "No")

        with st.expander("Inspect retrieval and evidence", expanded=True):
            retrieval_column, evidence_column = st.columns(2)

            with retrieval_column:
                st.markdown("**Initial retrieval ranking**")
                st.code(
                    "\n".join(
                        f"{rank}. {chunk_id}"
                        for rank, chunk_id in enumerate(
                            case["retrieved"],
                            start=1,
                        )
                    )
                )

            with evidence_column:
                st.markdown("**Reranked evidence**")
                st.code(
                    "\n".join(
                        f"{rank}. {chunk_id}"
                        for rank, chunk_id in enumerate(
                            case["evidence"],
                            start=1,
                        )
                    )
                )

            st.markdown("**Validated citations**")

            if case["citations"]:
                st.code("\n".join(case["citations"]))
            else:
                st.write("No citation required for a safe refusal.")

        st.subheader("Tesla T4 latency")

        latency_rows = [
            {
                "Pipeline stage": stage,
                "Latency (ms)": latency,
            }
            for stage, latency in case["latency"].items()
        ]

        st.dataframe(
            latency_rows,
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "Latency values were recorded during the verified Colab "
            "Tesla T4 evaluation and are not measured in this browser session."
        )

st.divider()
st.caption(
    "Portfolio engineering prototype using synthetic policy documents. "
    "The full reproducible GPU evaluation is available in the project notebook."
)