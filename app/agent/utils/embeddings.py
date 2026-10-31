import os
import hashlib
import math
from functools import lru_cache
from typing import List
import httpx
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.logger import logger
from app.core.config import settings

load_dotenv()

EXPECTED_EMBEDDING_DIM = int(os.getenv("GOOGLE_EMBEDDING_DIM", "768"))

NIM_URL = "https://integrate.api.nvidia.com/v1/embeddings"
NIM_BATCH = 64
# The model allows keeping its first 1024 or 512 values. 512 fits the vector(768) column with no
# schema change: the unused values stay zero, which cosine similarity ignores.
NIM_KEEP_DIMS = 512
NIM_TIMEOUT_SECONDS = 30
NIM_MAX_CHARS = 8000  # plenty to place a long answer, and the service refuses what is far longer

_EMBEDDING_MODEL_CANDIDATES = tuple(
    model_name
    for model_name in (
        os.getenv("GOOGLE_EMBEDDING_MODEL"),
        "models/gemini-embedding-001",
        "models/embedding-001",
    )
    if model_name
)


@lru_cache(maxsize=4)
def _get_embedding_model(model_name: str):
    return GoogleGenerativeAIEmbeddings(model=model_name)


def _shape_embedding(vector: List[float]) -> List[float]:
    values = [float(value) for value in (vector or [])]

    if len(values) > EXPECTED_EMBEDDING_DIM:
        logger.warning(
            "Embedding dimension mismatch: got %s values, trimming to %s",
            len(values),
            EXPECTED_EMBEDDING_DIM,
        )
        values = values[:EXPECTED_EMBEDDING_DIM]
    elif len(values) < EXPECTED_EMBEDDING_DIM:
        values.extend([0.0] * (EXPECTED_EMBEDDING_DIM - len(values)))

    magnitude = math.sqrt(sum(value * value for value in values))
    if magnitude > 1e-12:
        values = [value / magnitude for value in values]

    return values


def _embed_query_with_model(model_name: str, text: str) -> List[float]:
    embeddings = _get_embedding_model(model_name)
    try:
        vector = embeddings.embed_query(text, output_dimensionality=EXPECTED_EMBEDDING_DIM)
    except TypeError:
        vector = embeddings.embed_query(text)
    return _shape_embedding(vector)


def _embed_documents_with_model(model_name: str, texts: List[str]) -> List[List[float]]:
    embeddings = _get_embedding_model(model_name)
    try:
        vectors = embeddings.embed_documents(
            texts,
            output_dimensionality=EXPECTED_EMBEDDING_DIM,
        )
    except TypeError:
        vectors = embeddings.embed_documents(texts)
    return [_shape_embedding(vector) for vector in vectors]


def _nim_transport() -> httpx.BaseTransport | None:
    return None  # httpx's default; tests replace it


def _nim_request(texts: List[str], input_type: str) -> List[List[float] | None]:
    """One call; a 400 splits the batch, so one oversized text can't lose its batchmates."""
    with httpx.Client(transport=_nim_transport(), timeout=NIM_TIMEOUT_SECONDS) as client:
        reply = client.post(
            NIM_URL,
            headers={"Authorization": f"Bearer {settings.NVIDIA_API_KEY}"},
            json={"model": settings.NIM_EMBEDDING_MODEL, "input": texts, "input_type": input_type,
                  "encoding_format": "float"},
        )
    if reply.status_code == 400:
        if len(texts) == 1:
            logger.warning("NIM refused a text as too long or invalid; it gets no embedding")
            return [None]
        middle = len(texts) // 2
        return _nim_request(texts[:middle], input_type) + _nim_request(texts[middle:], input_type)
    if reply.status_code != 200:
        raise RuntimeError(f"NIM embeddings answered HTTP {reply.status_code}")
    rows = sorted(reply.json()["data"], key=lambda row: row["index"])
    return [_shape_embedding(row["embedding"][:NIM_KEEP_DIMS]) for row in rows]


def _embed_with_nim(texts: List[str], input_type: str) -> List[List[float]]:
    """`input_type` matters: this model embeds questions and stored text differently, and scoring
    one against the other without the flag is close to random."""
    if not settings.NVIDIA_API_KEY:
        raise RuntimeError("NVIDIA_API_KEY is not set")
    out: List[List[float]] = [[0.0] * EXPECTED_EMBEDDING_DIM for _ in texts]  # blank text has no meaning to embed
    filled = [i for i, text in enumerate(texts) if (text or "").strip()]
    for start in range(0, len(filled), NIM_BATCH):
        part = filled[start:start + NIM_BATCH]
        for i, vector in zip(part, _nim_request([texts[i][:NIM_MAX_CHARS] for i in part], input_type)):
            if vector is not None:
                out[i] = vector
    return out


def _embed_query(text: str) -> List[float]:
    if settings.EMBEDDING_PROVIDER == "nim":
        return _embed_with_nim([text], "query")[0]
    last_error: Exception | None = None
    for model_name in _EMBEDDING_MODEL_CANDIDATES:
        try:
            return _embed_query_with_model(model_name, text)
        except Exception as exc:
            last_error = exc
            logger.warning(f"Embedding model '{model_name}' failed: {exc}")

    raise RuntimeError(f"All embedding model candidates failed: {last_error}")


def _embed_documents(texts: List[str]) -> List[List[float]]:
    if settings.EMBEDDING_PROVIDER == "nim":
        return _embed_with_nim(texts, "passage")
    last_error: Exception | None = None
    for model_name in _EMBEDDING_MODEL_CANDIDATES:
        try:
            return _embed_documents_with_model(model_name, texts)
        except Exception as exc:
            last_error = exc
            logger.warning(f"Embedding model '{model_name}' failed for batch: {exc}")

    raise RuntimeError(f"All embedding model candidates failed: {last_error}")

def _fake_embedding(text: str) -> List[float]:
    """Deterministic unit vector derived from sha256(text); no network."""
    seed = hashlib.sha256((text or "").encode("utf-8")).digest()
    raw: List[float] = []
    counter = 0
    while len(raw) < EXPECTED_EMBEDDING_DIM:
        block = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        raw.extend((b - 127.5) / 127.5 for b in block)
        counter += 1
    return _shape_embedding(raw[:EXPECTED_EMBEDDING_DIM])


def get_embedding(text: str) -> List[float]:
    """Generates embedding for a single string using Gemini."""
    if settings.ENGINE_FAKE_LLM:
        return _fake_embedding(text)
    try:
        return _embed_query(text)
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return [0.0] * EXPECTED_EMBEDDING_DIM

def get_passage_embedding(text: str) -> List[float]:
    """One text to be stored and searched later (get_embedding is for what is searched with)."""
    return get_embeddings([text])[0]


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generates embeddings for a list of strings using Gemini."""
    if settings.ENGINE_FAKE_LLM:
        return [_fake_embedding(t) for t in texts]
    try:
        if not texts:
            return []

        return _embed_documents(texts)
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {e}")
        return [[0.0] * EXPECTED_EMBEDDING_DIM for _ in texts]
