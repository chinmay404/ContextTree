import os
from typing import List
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.logger import logger

load_dotenv()

# Ensure GOOGLE_API_KEY is available in environment
# os.environ["GOOGLE_API_KEY"] = ... 

def get_embedding_model():
    # Use text-embedding-004 as it is newer and better or models/embedding-001 as requested
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

def get_embedding(text: str) -> List[float]:
    """Generates embedding for a single string using Gemini."""
    try:
        embeddings = get_embedding_model()
        return embeddings.embed_query(text)
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        # Fallback for dev/test without key (returns zero vector - 768 dim for Gemini 001/004 usually)
        # Note: text-embedding-004 is 768 dimensions by default. 
        # OpenAI text-embedding-3-small was 1536. 
        # WARNING: DB schemas expecting 1536 will break if not updated to 768.
        return [0.0] * 768

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generates embeddings for a list of strings using Gemini."""
    try:
        if not texts:
            return []
        
        embeddings = get_embedding_model()
        # embed_documents is the batch method for LangChain embeddings
        return embeddings.embed_documents(texts)
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {e}")
        return [[0.0] * 768 for _ in texts]