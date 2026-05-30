import hashlib
import json
import os
import time
import uuid
from io import BytesIO

import chromadb
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


# -----------------------------
# Config
# -----------------------------
load_dotenv()
EMBEDDING_MODEL = "text-embedding-3-small"
ANSWER_MODEL = "gpt-4.1-mini"
JUDGE_MODEL = "gpt-4.1"  # stronger + different from answerer to reduce self-preference bias
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "rag_documents"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=3)


def get_collection():
    if "chroma_client" not in st.session_state:
        st.session_state.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    if "collection" not in st.session_state:
        st.session_state.collection = st.session_state.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # match text-embedding-3-small
        )
    return st.session_state.collection


# -----------------------------
# Helpers
# -----------------------------
def file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:16]


def extract_text(file_bytes: bytes, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".txt") or name.endswith(".md"):
        return file_bytes.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        reader = PdfReader(BytesIO(file_bytes))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    raise ValueError("Only PDF, TXT, and MD files are supported.")


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = text.replace("\x00", " ").strip()
    if overlap >= chunk_size:
        overlap = chunk_size // 2  # guard against infinite loop
    step = chunk_size - overlap

    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def already_indexed(collection, doc_hash: str) -> bool:
    existing = collection.get(where={"doc_hash": doc_hash}, limit=1)
    return len(existing["ids"]) > 0


def add_document_to_chroma(collection, filename: str, doc_hash: str, chunks: list[str]) -> int:
    embeddings = embed_texts(chunks)
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {"source": filename, "doc_hash": doc_hash, "chunk_index": i}
        for i in range(len(chunks))
    ]
    collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
    return len(chunks)


def retrieve(collection, question: str, top_k: int):
    question_embedding = embed_texts([question])[0]
    return collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )


def answer_question(question: str, results) -> str:
    docs = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_blocks = []
    for i, doc in enumerate(docs):
        source = metadatas[i].get("source", "unknown")
        chunk_index = metadatas[i].get("chunk_index", "?")
        context_blocks.append(f"Source: {source} | Chunk: {chunk_index}\n{doc}")

    context = "\n---\n".join(context_blocks)

    prompt = f"""
You are a RAG assistant.
Answer only using the provided context.
If the answer is not in the context, say: "I don't know based on the provided documents."

Context:
{context}

Question:
{question}
""".strip()

    response = client.responses.create(model=ANSWER_MODEL, input=prompt)
    return response.output_text


def safe_json_loads(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None

def retrieval_metrics(retrieved_ids: list[str], gold_id: str) -> dict:
    hit = 1 if gold_id in retrieved_ids else 0
    if hit:
        rank = retrieved_ids.index(gold_id) + 1  # 1-indexed
        mrr = 1 / rank
    else:
        rank = None
        mrr = 0.0
    return {"hit": hit, "mrr": round(mrr, 3), "rank": rank}


def generate_eval_set(collection, num_questions: int = 5) -> list[dict]:
    data = collection.get(limit=num_questions, include=["documents", "metadatas"])
    eval_items = []

    for chunk_id, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        prompt = f"""
Create one RAG evaluation question from this document chunk.
Return only valid JSON. Do not use markdown.

JSON format:
{{
  "question": "one clear question answerable only from the chunk",
  "expected_answer": "short answer based only on the chunk"
}}

Chunk:
{doc}
""".strip()

        try:
            response = client.responses.create(model=ANSWER_MODEL, input=prompt)
            item = safe_json_loads(response.output_text)
        except Exception as e:
            st.warning(f"Eval generation failed for chunk {meta.get('chunk_index', '?')}: {e}")
            continue

        if item and "question" in item and "expected_answer" in item:
            item["source"] = meta.get("source", "unknown")
            item["chunk_index"] = meta.get("chunk_index", "?")
            item["gold_chunk_id"] = chunk_id  # ← new: the Chroma UUID
            eval_items.append(item)
        else:
            st.warning(f"Could not parse eval question for chunk {meta.get('chunk_index', '?')}")

    return eval_items


def judge_answer(question: str, expected_answer: str, actual_answer: str, retrieved_context: str) -> dict:
    prompt = f"""
You are evaluating a RAG system.
Return only valid JSON.
Score from 0 to 1:
- correctness: does the actual answer match the expected answer?
- faithfulness: is the actual answer supported by the retrieved context?
- retrieval_quality: does the retrieved context contain the information needed?

Question:
{question}

Expected answer:
{expected_answer}

Actual answer:
{actual_answer}

Retrieved context:
{retrieved_context}

JSON format:
{{
  "correctness": 0.0,
  "faithfulness": 0.0,
  "retrieval_quality": 0.0,
  "feedback": "short feedback"
}}
""".strip()

    try:
        response = client.responses.create(model=JUDGE_MODEL, input=prompt)
        item = safe_json_loads(response.output_text)
    except Exception as e:
        return {
            "correctness": 0.0, "faithfulness": 0.0, "retrieval_quality": 0.0,
            "feedback": f"Judge call failed: {e}",
        }

    if item:
        return item
    return {
        "correctness": 0.0, "faithfulness": 0.0, "retrieval_quality": 0.0,
        "feedback": "Judge returned invalid JSON.",
    }


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="RAG Playground", layout="wide")
st.title("RAG Playground")
st.caption("Upload documents, index them, ask questions, and inspect retrieval quality.")

collection = get_collection()

if "last_index_message" in st.session_state:
    st.success(st.session_state.pop("last_index_message"))

with st.sidebar:
    st.header("Settings")
    st.metric("Chunks in vector DB", collection.count())
    chunk_size = st.slider("Chunk size", 300, 2000, 800, step=100)
    overlap = st.slider("Chunk overlap", 0, 500, 100, step=50)
    top_k = st.slider("Top K chunks", 1, 10, 4)

    st.divider()
    if st.button("Clear vector database"):
        st.session_state.chroma_client.delete_collection(COLLECTION_NAME)
        st.session_state.pop("collection", None)
        st.success("Vector database cleared.")
        st.rerun()

st.subheader("1. Upload documents")
uploaded_files = st.file_uploader(
    "Upload PDF, TXT, or MD files",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
)

if uploaded_files and st.button("Index documents"):
    total_chunks = 0
    skipped = 0
    with st.spinner("Indexing documents..."):
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.read()
            doc_hash = file_hash(file_bytes)

            if already_indexed(collection, doc_hash):
                st.info(f"{uploaded_file.name}: already indexed, skipping.")
                skipped += 1
                continue

            try:
                text = extract_text(file_bytes, uploaded_file.name)
            except Exception as e:
                st.error(f"{uploaded_file.name}: extraction failed ({e})")
                continue

            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            st.write(f"{uploaded_file.name}: {len(text)} chars, {len(chunks)} chunks")

            if chunks:
                total_chunks += add_document_to_chroma(collection, uploaded_file.name, doc_hash, chunks)

    if total_chunks > 0:
        st.session_state["last_index_message"] = f"Indexed {total_chunks} chunks ({skipped} files skipped)."
        st.rerun()
    elif skipped == 0:
        st.error("No chunks were created. Try a TXT file or a non-scanned PDF.")

st.divider()
st.subheader("2. Ask a question")
question = st.text_input("Question")

if question:
    if collection.count() == 0:
        st.error("No chunks indexed yet. Upload a document first.")
        st.stop()

    # Only re-run retrieval if the question changed
    if st.session_state.get("last_question") != question:
        start_time = time.time()
        results = retrieve(collection, question, top_k=top_k)
        answer = answer_question(question, results)
        latency = time.time() - start_time

        st.session_state["last_question"] = question
        st.session_state["last_results"] = results
        st.session_state["last_answer"] = answer
        st.session_state["last_latency"] = latency
        st.session_state.pop("manual_eval_result", None)  # reset previous eval

    results = st.session_state["last_results"]
    answer = st.session_state["last_answer"]
    latency = st.session_state["last_latency"]

    st.markdown("### Answer")
    st.write(answer)

    st.markdown("### Basic stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Top K", top_k)
    col2.metric("Latency", f"{latency:.2f}s")
    col3.metric("Retrieved chunks", len(results["documents"][0]))

    st.markdown("### Retrieved context")
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        with st.expander(f"Chunk {i + 1} | {meta.get('source')} | distance: {distance:.4f}"):
            st.write(doc)

    # --- Manual evaluation ---
    st.markdown("### Evaluate this answer")
    expected = st.text_area(
        "Expected answer (what the correct answer should be)",
        key="manual_expected",
    )

    if st.button("Evaluate this answer"):
        if not expected.strip():
            st.warning("Provide an expected answer first.")
        else:
            retrieved_context = "\n".join(results["documents"][0])
            with st.spinner("Judging..."):
                scores = judge_answer(question, expected, answer, retrieved_context)
            st.session_state["manual_eval_result"] = scores

    if "manual_eval_result" in st.session_state:
        scores = st.session_state["manual_eval_result"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Correctness", scores.get("correctness", 0))
        c2.metric("Faithfulness", scores.get("faithfulness", 0))
        c3.metric("Retrieval quality", scores.get("retrieval_quality", 0))
        st.caption(f"Feedback: {scores.get('feedback', '')}")

st.divider()
st.subheader("3. Dynamic RAG evaluation")

num_eval_questions = st.slider("Number of eval questions", 1, 10, 5)

if st.button("Generate eval set from documents"):
    if collection.count() == 0:
        st.error("Index documents first.")
    else:
        with st.spinner("Generating eval questions..."):
            st.session_state["eval_set"] = generate_eval_set(collection, num_eval_questions)
        st.success(f"Generated {len(st.session_state['eval_set'])} eval questions.")

if "eval_set" in st.session_state:
    st.markdown("### Eval set")
    st.dataframe(pd.DataFrame(st.session_state["eval_set"]))

    if st.button("Run evaluation"):
        rows = []
        with st.spinner("Running RAG evaluation..."):
            for item in st.session_state["eval_set"]:
                q = item["question"]
                expected = item["expected_answer"]
                gold_id = item.get("gold_chunk_id")

                start_time = time.time()
                results = retrieve(collection, q, top_k=top_k)
                actual = answer_question(q, results)
                latency = time.time() - start_time

                retrieved_ids = results["ids"][0]
                retrieved_context = "\n".join(results["documents"][0])
                scores = judge_answer(q, expected, actual, retrieved_context)
                ret_metrics = retrieval_metrics(retrieved_ids, gold_id)

                rows.append({
                    "question": q,
                    "expected_answer": expected,
                    "actual_answer": actual,
                    "hit@k": ret_metrics["hit"],
                    "mrr": ret_metrics["mrr"],
                    "rank": ret_metrics["rank"],
                    "correctness": scores.get("correctness", 0),
                    "faithfulness": scores.get("faithfulness", 0),
                    "retrieval_quality": scores.get("retrieval_quality", 0),
                    "latency": round(latency, 2),
                    "feedback": scores.get("feedback", ""),
            })

        st.session_state["eval_results"] = pd.DataFrame(rows)

if "eval_results" in st.session_state:
    st.markdown("### Evaluation results")
    st.dataframe(st.session_state["eval_results"])

    st.markdown("### Average scores")
    col1, col2, col3, col4, col5 = st.columns(5)
    df = st.session_state["eval_results"]
    col1.metric(f"Hit@{top_k}", f"{df['hit@k'].mean():.0%}")
    col2.metric("MRR", round(df["mrr"].mean(), 2))
    col3.metric("Correctness", round(df["correctness"].mean(), 2))
    col4.metric("Faithfulness", round(df["faithfulness"].mean(), 2))
    col5.metric("Retrieval quality", round(df["retrieval_quality"].mean(), 2))


    # python -m streamlit run playground.py