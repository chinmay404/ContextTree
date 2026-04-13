import os
import math
from functools import lru_cache
from typing import List
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.logger import logger

load_dotenv()

EXPECTED_EMBEDDING_DIM = int(os.getenv("GOOGLE_EMBEDDING_DIM", "768"))

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


def _embed_query(text: str) -> List[float]:
    last_error: Exception | None = None
    for model_name in _EMBEDDING_MODEL_CANDIDATES:
        try:
            return _embed_query_with_model(model_name, text)
        except Exception as exc:
            last_error = exc
            logger.warning(f"Embedding model '{model_name}' failed: {exc}")

    raise RuntimeError(f"All embedding model candidates failed: {last_error}")


def _embed_documents(texts: List[str]) -> List[List[float]]:
    last_error: Exception | None = None
    for model_name in _EMBEDDING_MODEL_CANDIDATES:
        try:
            return _embed_documents_with_model(model_name, texts)
        except Exception as exc:
            last_error = exc
            logger.warning(f"Embedding model '{model_name}' failed for batch: {exc}")

    raise RuntimeError(f"All embedding model candidates failed: {last_error}")

def get_embedding(text: str) -> List[float]:
    """Generates embedding for a single string using Gemini."""
    try:
        return _embed_query(text)
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return [0.0] * EXPECTED_EMBEDDING_DIM

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generates embeddings for a list of strings using Gemini."""
    try:
        if not texts:
            return []

        return _embed_documents(texts)
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {e}")
        return [[0.0] * EXPECTED_EMBEDDING_DIM for _ in texts]
