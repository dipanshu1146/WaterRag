
import re
from collections import Counter
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
from src.config import VECTOR_DB_DIR as DB_DIR  # single shared path
COLLECTION_NAME = "wastewater_documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def normalize_key(key):
    """Remove spaces and punctuation from a field name."""
    return re.sub(r"[^a-z0-9]", "", key.lower())


def inspect_table_rows():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(DB_DIR),
    )

    collection = vectorstore._collection

    # Retrieve all indexed table rows, not just the top 5.
    data = collection.get(
        where={"content_type": "table_row"},
        include=["documents", "metadatas"],
    )

    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []

    compliance_counts = Counter()
    operational_counts = Counter()
    missing_compliance = 0
    missing_operational = 0

    for document in documents:
        compliance_value = None
        operational_value = None

        for line in (document or "").splitlines():
            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            normalized = normalize_key(key)
            value = value.strip()

            if "compliancestatus" in normalized:
                compliance_value = value or "(empty)"

            elif (
                "statusofstp" in normalized
                or "operationalstatus" in normalized
            ):
                operational_value = value or "(empty)"

        if compliance_value is None:
            missing_compliance += 1
        else:
            compliance_counts[compliance_value] += 1

        if operational_value is None:
            missing_operational += 1
        else:
            operational_counts[operational_value] += 1

    print("\n========== INDEX SUMMARY ==========")
    print(f"Total vectors in database: {collection.count()}")
    print(f"Indexed table rows found: {len(documents)}")

    print("\n========== COMPLIANCE VALUES ==========")
    for status, count in compliance_counts.most_common():
        print(f"{status}: {count}")

    print(f"Rows without compliance field: {missing_compliance}")

    print("\n========== OPERATIONAL STATUS VALUES ==========")
    for status, count in operational_counts.most_common():
        print(f"{status}: {count}")

    print(f"Rows without operational field: {missing_operational}")

    print("\n========== SAMPLE SOURCE RECORDS ==========")
    shown = 0

    for document, metadata in zip(documents, metadatas):
        if (
            "compliance" in (document or "").lower()
            or "status of stp" in (document or "").lower()
        ):
            print("\n------------------------------------")
            print("Source:", metadata.get("source", "Unknown"))
            print("Page:", metadata.get("page", "N/A"))
            print(document)
            shown += 1

            if shown == 5:
                break


if __name__ == "__main__":
    inspect_table_rows()
