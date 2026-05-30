import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import streamlit as st
from core.logs import clear_logs, load_logs

st.set_page_config(page_title="Logs", layout="wide")
st.title("Test Logs")
st.caption("Every Q&A run recorded — use this to compare settings, reranking on/off, and answer quality over time.")

logs = load_logs()

if not logs:
    st.info("No logs yet. Ask some questions in the Playground first.")
else:
    df = pd.DataFrame(logs)

    # Sidebar filters
    with st.sidebar:
        st.header("Filters")
        filter_rerank = st.multiselect(
            "Reranking", options=[True, False], default=[True, False],
            format_func=lambda x: "ON" if x else "OFF",
        )
        if "sources" in df.columns:
            all_sources = sorted({s for row in df["sources"] for s in (row if isinstance(row, list) else [])})
            filter_sources = st.multiselect("Source document", options=all_sources, default=all_sources)

    if filter_rerank is not None:
        df = df[df["rerank"].isin(filter_rerank)]
    if "sources" in df.columns and filter_sources:
        df = df[df["sources"].apply(lambda row: any(s in filter_sources for s in (row if isinstance(row, list) else [])))]

    # Column order
    preferred = ["timestamp", "question", "chunk_size", "overlap", "top_k",
                 "rerank", "rerank_candidates", "latency",
                 "correctness", "faithfulness", "retrieval_quality",
                 "sources", "answer", "feedback"]
    cols = [c for c in preferred if c in df.columns]
    display_df = df[cols].sort_values("timestamp", ascending=False).fillna("N/A").astype(str)
    st.dataframe(display_df, width="stretch", height=400)

    # Summary metrics (only rows that have been evaluated)
    evaluated = df.dropna(subset=["correctness", "faithfulness", "retrieval_quality"])
    if not evaluated.empty:
        st.divider()
        st.subheader("Average scores (evaluated runs only)")

        groups = evaluated.groupby("rerank")[["correctness", "faithfulness", "retrieval_quality", "latency"]].mean().round(3)
        groups.index = groups.index.map(lambda x: "Rerank ON" if x else "Rerank OFF")
        st.dataframe(groups, width="stretch")

    st.divider()
    if st.button("Clear all logs", type="secondary"):
        clear_logs()
        st.success("Logs cleared.")
        st.rerun()
