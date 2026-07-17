
import io
import logging
import tempfile
import os
from typing import List, Optional, Tuple
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel

from app.api.auth import get_current_user_id
from langchain_community.document_loaders import CSVLoader, Docx2txtLoader, PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.agent.store.PostgresStore import PostgresConversationStore
from app.agent.utils.embeddings import get_embeddings
from app.core.logger import logger

router = APIRouter()

class IngestRequest(BaseModel):
    file_id: str
    node_id: Optional[str] = None
    canvas_id: Optional[str] = None
    # Identity comes from the verified service JWT (app/api/auth.py),
    # never from the request body.

def load_and_split_documents(data: bytes, mime_type: str, file_name: str) -> Tuple[List[str], List[dict], str, Optional[str]]:
    """
    Returns chunks, metadatas, full_text (for preview), and error message if any.
    """
    import pathlib

    # Allowlist (owner decision 2026-07-17): documents only. Anything else
    # errors loudly instead of falling back to TextLoader garbage — the
    # error lands on the node via update_node_processing_error.
    lower = (file_name or "").lower()
    if mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or lower.endswith(".docx"):
        suffix = ".docx"
    elif mime_type == "application/pdf" or lower.endswith(".pdf"):
        suffix = ".pdf"
    elif mime_type == "text/markdown" or lower.endswith(".md"):
        suffix = ".md"
    elif mime_type == "text/plain" or lower.endswith(".txt"):
        suffix = ".txt"
    elif lower.endswith(".doc"):
        raise ValueError(
            "Legacy .doc format can't be parsed — save it as .docx or PDF and re-upload."
        )
    else:
        ext = pathlib.Path(lower).suffix or "(none)"
        raise ValueError(
            f"Unsupported file type {ext}. Allowed: PDF, TXT, MD, DOCX."
        )

    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        docs = []
        if suffix == ".docx":
            loader = Docx2txtLoader(tmp_path)
            docs = loader.load()
        elif suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
        elif suffix == ".md":
             # Use TextLoader for MD if unstructured not available, but user didn't ask for unstructured specifically
             # TextLoader works for MD
             loader = TextLoader(tmp_path)
             docs = loader.load()
        else:
            loader = TextLoader(tmp_path)
            docs = loader.load()
            
        # Split
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        
        chunks = [d.page_content for d in splits]
        metadatas = [d.metadata for d in splits]
        
        # Combine text for preview (simple join)
        # For CSV, page_content is row representation
        full_text = "\n\n".join([d.page_content for d in docs])
        
        return chunks, metadatas, full_text, None
        
    except Exception as e:
        logger.error(f"Error loading file {file_name} with langchain: {e}")
        return [], [], "", "Failed to read file content"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass

def process_file_background(request: IngestRequest):
    """
    1. Fetch binary from DB
    2. Extract Text via Langchain Loaders
    3. Split
    4. Generate Embeddings
    5. Save chunks with metadata
    """
    logger.info(f"Starting background ingestion for file {request.file_id}")
    store = PostgresConversationStore()
    
    # 1. Fetch
    file_record = store.get_file_binary(request.file_id)
    if not file_record:
        logger.error(f"File {request.file_id} not found in DB")
        if request.node_id:
            store.update_node_processing_error(
                request.node_id, "File not found in storage"
            )
        return

    data = file_record.get('data')
    # Prioritize mime_type from DB as requested for robustness
    mime_type = file_record.get('mime_type')
    file_name = file_record.get('file_name', '')

    logger.info(f"Processing file {request.file_id}: name={file_name}, mime={mime_type}")

    if not data:
        logger.error(f"No binary data for file {request.file_id}")
        if request.node_id:
            store.update_node_processing_error(
                request.node_id, "File content missing"
            )
        return
        
    # Convert memoryview to bytes if needed
    if isinstance(data, memoryview):
        data = data.tobytes()

    # 2. Parse & Chunk
    chunks, metadatas, text_content, load_error = load_and_split_documents(data, mime_type, file_name)

    if load_error:
        logger.warning(f"File processing error for {request.file_id}: {load_error}")
        if request.node_id:
            store.update_node_processing_error(request.node_id, load_error)
        return

    if not chunks:
        logger.warning(f"No chunks extracted from file {request.file_id}")
        if request.node_id:
            limit_content = text_content[:100000] if text_content else ""
            if limit_content:
                store.update_node_data_content(request.node_id, limit_content)
            else:
                store.update_node_processing_error(
                    request.node_id, "No text content extracted"
                )
        return

    # 4. Embed
    logger.info(f"Generating embeddings for {len(chunks)} chunks")
    try:
        embeddings = get_embeddings(chunks)
    except Exception as e:
        logger.error(f"Embedding generation failed for {request.file_id}: {e}")
        if request.node_id:
            store.update_node_processing_error(
                request.node_id, "Failed to generate embeddings"
            )
        return

    # 5. Save
    try:
        store.save_file_chunks(request.file_id, chunks, embeddings, metadatas)
    except Exception as e:
        logger.error(f"Saving file chunks failed for {request.file_id}: {e}")
        if request.node_id:
            store.update_node_processing_error(
                request.node_id, "Failed to save processed file"
            )
        return
    
    # 6. Update Node with extracted content
    if request.node_id:
        limit_content = text_content[:100000] # 100KB limit for immediate preview
        store.update_node_data_content(request.node_id, limit_content)
        
    logger.info(f"Successfully ingested file {request.file_id}")


@router.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
async def ingest_file(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
):
    """
    Triggers background ingestion of a file already stored in DB.
    """
    logger.info(f"Ingest request received for file_id: {request.file_id} user: {user_id}")
    background_tasks.add_task(process_file_background, request)
    return {"status": "processing_started", "file_id": request.file_id}
