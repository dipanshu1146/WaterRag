
import hashlib
import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.load_documents import load_documents


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
from src.config import VECTOR_DB_DIR as DB_DIR  # single shared path
COLLECTION_NAME = "wastewater_documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def split_documents(documents):
    """Split narrative text while keeping table rows intact."""
    text_documents = []
    table_documents = []

    for document in documents:
        content_type = document.metadata.get("content_type", "text")

        if content_type == "table_row":
            # Preserve each row so its labels and values stay together.
            table_documents.append(document)
        else:
            text_documents.append(document)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )

    text_chunks = splitter.split_documents(text_documents)

    return text_chunks + table_documents


def make_document_id(document):
    """Generate a stable ID for each chunk."""
    identity = {
        "source": document.metadata.get("source", ""),
        "page": document.metadata.get("page", ""),
        "table_number": document.metadata.get("table_number", ""),
        "content_type": document.metadata.get("content_type", "text"),
        "content": document.page_content,
    }

    identity_string = json.dumps(
        identity,
        sort_keys=True,
        ensure_ascii=False,
    )

    return hashlib.sha256(identity_string.encode("utf-8")).hexdigest()


def clean_metadata(metadata):
    """Keep only metadata types supported by ChromaDB."""
    cleaned = {}

    for key, value in metadata.items():
        if value is None:
            continue

        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)

    return cleaned


def build_index():
    print("Loading extracted wastewater records...")

    records = load_documents()

    if not records:
        print("No records found. Check data/txt and data/uploads.")
        return

    print(f"Records loaded: {len(records)}")

    chunks = split_documents(records)
    print(f"Chunks prepared for indexing: {len(chunks)}")

    # Clean metadata and calculate stable IDs.
    for chunk in chunks:
        chunk.metadata = clean_metadata(chunk.metadata)

    ids = [make_document_id(chunk) for chunk in chunks]

    if len(ids) != len(set(ids)):
        print("Warning: duplicate chunk IDs detected.")
        unique_chunks = {}
        for chunk, document_id in zip(chunks, ids):
            unique_chunks[document_id] = chunk

        ids = list(unique_chunks.keys())
        chunks = list(unique_chunks.values())

    print("Loading local embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print("Opening ChromaDB vector store...")

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(DB_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )

    print("Adding documents to the vector index...")

    # Stable IDs allow existing chunks to be updated on reruns.
    vectorstore.add_documents(
        documents=chunks,
        ids=ids,
    )

    print("\nIndexing completed successfully.")
    print(f"Vector database: {DB_DIR}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Chunks processed: {len(chunks)}")
    print(f"Vectors stored: {vectorstore._collection.count()}")


if __name__ == "__main__":
    build_index()
