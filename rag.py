import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
from openai import OpenAI

DATA_DIR = Path("data")
INDEX_FILE = DATA_DIR / "index.json"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1400, overlap: int = 250):
    text = clean_text(text)
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _client():
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def embed(texts):
    response = _client().embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def save_index(chunks, metadata):
    DATA_DIR.mkdir(exist_ok=True)
    vectors = []
    for i in range(0, len(chunks), 100):
        vectors.extend(embed(chunks[i:i + 100]))
    records = []
    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        records.append({
            "id": hashlib.sha1(chunk.encode()).hexdigest()[:12],
            "chunk_index": idx,
            "text": chunk,
            "embedding": vector,
            **metadata,
        })
    INDEX_FILE.write_text(json.dumps(records), encoding="utf-8")
    return len(records)


def load_index():
    if not INDEX_FILE.exists():
        return []
    return json.loads(INDEX_FILE.read_text(encoding="utf-8"))


def retrieve(question: str, top_k: int = 5):
    records = load_index()
    if not records:
        return []
    q = np.asarray(embed([question])[0], dtype=float)
    q /= np.linalg.norm(q) + 1e-12
    scored = []
    for record in records:
        v = np.asarray(record["embedding"], dtype=float)
        v /= np.linalg.norm(v) + 1e-12
        scored.append((float(np.dot(q, v)), record))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"score": score, **record} for score, record in scored[:top_k]]


def answer_question(question: str, top_k: int = 5):
    sources = retrieve(question, top_k)
    if not sources:
        return "No filing has been indexed yet.", []

    context = "\n\n".join(
        f"[Source {i}] {s['text']}" for i, s in enumerate(sources, start=1)
    )
    prompt = f"""You are a financial research assistant.
Answer the question using ONLY the supplied SEC filing excerpts.
If the excerpts do not support an answer, say that the filing excerpts do not contain enough information.
Cite factual claims using [Source 1], [Source 2], etc.

QUESTION:
{question}

FILING EXCERPTS:
{context}
"""
    response = _client().responses.create(model=CHAT_MODEL, input=prompt)
    return response.output_text, sources
