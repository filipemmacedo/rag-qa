# RAG Playground

Built by **Filipe Macedo** as a hands-on way to explore RAG internals — chunk sizes, overlap, retrieval strategies, evaluation metrics — and to share that learning with anyone else who wants to dig into how RAG actually works under the hood.

If you're here to learn, experiment, or just break things to see what happens: welcome.

---

A Streamlit app for building and evaluating a Retrieval-Augmented Generation (RAG) pipeline. Upload documents, ask questions against them, and benchmark retrieval and answer quality — all in a browser UI.

## Features

- **Document ingestion** — Upload PDF, TXT, or MD files. Text is chunked and embedded into a persistent ChromaDB vector store. Duplicate documents are detected by SHA-256 hash and skipped automatically.
- **Question answering** — Retrieves the top-K most relevant chunks via cosine similarity and generates an answer using GPT-4.1-mini, grounded strictly in the retrieved context.
- **Manual evaluation** — Provide an expected answer and a judge model (GPT-4.1) scores the response on correctness, faithfulness, and retrieval quality.
- **Automated eval set** — Generate question/answer pairs from indexed chunks and run a full evaluation sweep with Hit@K, MRR, correctness, faithfulness, and retrieval quality metrics.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

### 3. Run the app

```bash
streamlit run playground.py
```

## Usage

1. **Upload documents** — drag and drop one or more PDF/TXT/MD files and click **Index documents**.
2. **Ask a question** — type a question; the app retrieves relevant chunks and displays the answer with latency and source attribution.
3. **Evaluate manually** — enter an expected answer and click **Evaluate this answer** to get LLM-judge scores.
4. **Run automated eval** — use the slider to set the number of eval questions, click **Generate eval set**, then **Run evaluation** to see a full metrics table.

## Configuration (sidebar)

| Setting | Default | Description |
|---|---|---|
| Chunk size | 800 | Characters per chunk |
| Chunk overlap | 100 | Overlap between consecutive chunks |
| Top K chunks | 4 | Number of chunks retrieved per query |

## Models

| Role | Model |
|---|---|
| Embeddings | `text-embedding-3-small` |
| Answer generation | `gpt-4.1-mini` |
| LLM judge | `gpt-4.1` |

## Project structure

```
playground.py      # Streamlit app (single file)
requirements.txt   # Python dependencies
.env               # API keys (not committed)
chroma_db/         # Persistent vector store (auto-created)
```
