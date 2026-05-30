import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime

LOGS_PATH = "./rag_logs.jsonl"


@dataclass
class LogEntry:
    timestamp: str
    question: str
    answer: str
    chunk_size: int
    overlap: int
    top_k: int
    latency: float
    rerank: bool
    rerank_candidates: int
    sources: list
    correctness: float | None = None
    faithfulness: float | None = None
    retrieval_quality: float | None = None
    feedback: str | None = None


def make_entry(question: str, answer: str, chunk_size: int, overlap: int,
               top_k: int, latency: float, rerank: bool,
               rerank_candidates: int, results: dict) -> LogEntry:
    sources = [m.get("source", "unknown") for m in results["metadatas"][0]]
    return LogEntry(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        question=question,
        answer=answer,
        chunk_size=chunk_size,
        overlap=overlap,
        top_k=top_k,
        latency=round(latency, 2),
        rerank=rerank,
        rerank_candidates=rerank_candidates,
        sources=sources,
    )


def append_log(entry: LogEntry) -> None:
    with open(LOGS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry)) + "\n")


def load_logs() -> list[dict]:
    if not os.path.exists(LOGS_PATH):
        return []
    entries = []
    with open(LOGS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def update_last_log_scores(question: str, scores: dict) -> None:
    entries = load_logs()
    for i in range(len(entries) - 1, -1, -1):
        if entries[i]["question"] == question:
            entries[i].update({
                "correctness": scores.get("correctness"),
                "faithfulness": scores.get("faithfulness"),
                "retrieval_quality": scores.get("retrieval_quality"),
                "feedback": scores.get("feedback"),
            })
            break
    with open(LOGS_PATH, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def clear_logs() -> None:
    if os.path.exists(LOGS_PATH):
        os.remove(LOGS_PATH)
