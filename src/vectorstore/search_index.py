import hashlib
from langchain_core.documents import Document
from src.ingestion.load_documents import load_excel_documents
import re
from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parents[2]
from src.config import VECTOR_DB_DIR as DB_DIR  # single shared path
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "wastewater_documents"


def create_vectorstore():
    # Hosted deployments use Supabase pgvector + Hugging Face's hosted
    # feature-extraction endpoint. Local development retains Chroma.
    from src.vectorstore.cloud_store import CloudVectorStore, cloud_mode_enabled

    if cloud_mode_enabled():
        return CloudVectorStore()

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(DB_DIR),
    )





def index_excel_documents():
    """Replace previously indexed Excel rows with corrected records."""
    documents = load_excel_documents()

    vectorstore = create_vectorstore()
    collection = vectorstore._collection

    # Delete only Excel records; preserve indexed PDF and other content.
    existing = collection.get(
        where={"file_type": "xlsx"},
        include=["metadatas"],
    )

    existing_ids = existing.get("ids", [])

    if existing_ids:
        collection.delete(ids=existing_ids)
        print(f"Removed old Excel records: {len(existing_ids)}")

    if not documents:
        print("No Excel records found to index.")
        return

    document_ids = []

    for document in documents:
        identity = (
            f"{document.metadata.get('source', '')}|"
            f"{document.metadata.get('sheet', '')}|"
            f"{document.metadata.get('row_number', '')}"
            f"{document.metadata.get('sample_type', '')}"
        )
        document_ids.append(
            hashlib.sha256(identity.encode("utf-8")).hexdigest()
        )

    vectorstore.add_documents(
        documents=documents,
        ids=document_ids,
    )

    print(f"Excel records indexed: {len(documents)}")





def normalize_source(source: str | Path) -> str:
    """Canonical absolute form of a stored ``source`` path.

    Relative sources (from bulk ingestion, e.g. ``data/uploads/x.pdf``) are
    resolved against the project root, NOT the process working directory.
    """
    path = Path(source)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def source_variants(source: str | Path) -> list[str]:
    """Every spelling a source may have been indexed under."""
    absolute = Path(normalize_source(source))
    variants = {str(absolute), str(Path(source))}
    # Cloud uploads use the unique stored filename as a stable source key.
    variants.add(Path(source).name)

    try:
        relative = absolute.relative_to(PROJECT_ROOT)
        variants.update({
            str(relative),
            relative.as_posix(),
            f"./{relative.as_posix()}",
            str(Path(str(relative).replace("/", "\\"))),
        })
    except ValueError:
        pass

    return sorted(variants)


def delete_source_records(source: str | Path, vectorstore=None) -> int:
    """Delete all Chroma records for a file, whether indexed with an absolute
    or a relative ``source`` path. Returns the number of records removed."""
    vectorstore = vectorstore or create_vectorstore()
    collection = vectorstore._collection

    matches = collection.get(
        where={"source": {"$in": source_variants(source)}},
        include=["metadatas"],
    )
    ids = matches.get("ids") or []

    for start in range(0, len(ids), 500):
        collection.delete(ids=ids[start:start + 500])

    return len(ids)


def index_uploaded_file(file_path: str | Path, vectorstore=None) -> int:
    """Index one uploaded file via the unified pipeline.

    extract -> format-aware chunking -> normalised metadata -> Chroma.

    Existing records for the same file are replaced, so calling this again
    (the reindex endpoint) is idempotent. Raises ``UnsupportedDocumentError``
    when the file yields no indexable content, so a caller can never report
    success for zero records.
    """
    from src.ingestion.pipeline import ingest_file

    path = Path(file_path).resolve()
    chunks = ingest_file(path)  # raises on unsupported / empty

    vectorstore = vectorstore or create_vectorstore()
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()

    from src.vectorstore.cloud_store import CloudVectorStore
    cloud_mode = isinstance(vectorstore, CloudVectorStore)
    if cloud_mode:
        # A stable source value survives Render restarts and maps back to the
        # object in the private Supabase Storage bucket.
        for chunk in chunks:
            chunk.metadata["source"] = path.name
            chunk.metadata["file_name"] = path.name

    documents, ids = [], []
    for position, chunk in enumerate(chunks):
        documents.append(Document(page_content=chunk.page_content, metadata=chunk.metadata))
        ids.append(hashlib.sha256(f"{file_hash}|{position}".encode("utf-8")).hexdigest())

    # Replace any earlier (possibly partial) index of this same file.
    delete_source_records(path.name if cloud_mode else path, vectorstore)

    for start in range(0, len(documents), 500):
        vectorstore.add_documents(
            documents=documents[start:start + 500],
            ids=ids[start:start + 500],
        )

    return len(documents)


def search_index(query: str, k: int = 5):
    """Prioritize exact STP IDs, then use semantic search."""
    vectorstore = create_vectorstore()
    collection = vectorstore._collection

    # 1. Look for an STP identifier such as YMN-STP-001.
    id_match = re.search(
        r"\b[A-Z]{2,5}-STP-\d{3,}\b",
        query.upper(),
    )

    if id_match:
        identifier = id_match.group(0)

        # Search indexed Excel records for the exact identifier.
        records = collection.get(
            where={"file_type": "xlsx"},
            include=["documents", "metadatas"],
        )

        exact_results = []

        for content, metadata in zip(
            records.get("documents", []),
            records.get("metadatas", []),
        ):
            if identifier in content.upper():
                document = Document(
                    page_content=content,
                    metadata=metadata,
                )
                exact_results.append((document, 0.0))

        if exact_results:
            results = exact_results[:k]
            town = None
            print(f"\nQuestion: {query}")
            print(f"Exact STP ID match: {identifier}")
            print(f"Results returned: {len(results)}")

            for rank, (document, score) in enumerate(
                results, start=1
            ):
                print("\n" + "=" * 50)
                print(f"Result {rank}")
                print("Match type: Exact identifier")
                print(f"Source: {document.metadata.get('source', 'Unknown')}")
                print(f"Row: {document.metadata.get('row_number', 'N/A')}")
                print("-" * 50)
                print(document.page_content)

            return results

        print(f"No exact Excel record found for {identifier}.")
        print("Trying semantic search instead.")

    # 2. Fall back to semantic search for general questions.
    results = vectorstore.similarity_search_with_score(
        query,
        k=k,
        filter={"content_type": "table_row"},
    )

    # 3. Extract a town name if one is explicitly mentioned.
    town_match = re.search(
        r"\b(?:of|for|in|at|about)\s+"
        r"([A-Za-z][A-Za-z -]*?)\s*(?:[?.!,]|$)",
        query,
        re.IGNORECASE,
    )

    town = town_match.group(1).strip() if town_match else None

    if town:
        exact_results = []

        for document, score in results:
            record_town = re.search(
                r"(?im)^Town:\s*(.+?)\s*$",
                document.page_content,
            )

            if (
                record_town
                and record_town.group(1).strip().casefold()
                == town.casefold()
            ):
                exact_results.append((document, score))

        results = exact_results

    if not results:
        print("No matching records found.")
        if town:
            print(f"No exact town match found for: {town}")
            print("Try using the town name exactly as it appears in the source.")
        return []

    print(f"\nQuestion: {query}")
    print(f"Results returned: {len(results)}")

    if town:
        print(f"Town filter: {town}")

    for rank, (document, score) in enumerate(results, start=1):
        print("\n" + "=" * 50)
        print(f"Result {rank}")
        print(f"Distance score: {score:.4f}")
        print(f"Source: {document.metadata.get('source', 'Unknown')}")
        print(f"Page: {document.metadata.get('page', 'N/A')}")
        print(f"Table: {document.metadata.get('table_number', 'N/A')}")
        print(f"Content type: {document.metadata.get('content_type', 'Unknown')}")
        print("-" * 50)
        print(document.page_content)

    return results



if __name__ == "__main__":
    question = input("Ask a wastewater treatment question: ").strip()

    if question:
        search_index(question)
    else:
        print("Please enter a question.")
