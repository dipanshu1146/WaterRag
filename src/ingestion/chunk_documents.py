
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Strategy 1: For normal paragraphs and long text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=75,
    separators=["\n\n", "\n", ". ", " ", ""]
)


def is_structured_text(text):
    """
    Check whether a text document mainly contains
    key-value fields, such as Company: ABC Industries.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if not lines:
        return False

    field_lines = sum(
        1 for line in lines
        if ":" in line and not line.startswith("http:")
    )

    # Treat text as structured if at least half its
    # non-empty lines look like key-value fields.
    return field_lines / len(lines) >= 0.5


def extract_fields(text):
    """Extract key-value fields from structured text."""
    fields = {}

    for line in text.splitlines():
        line = line.strip()

        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()

    return fields


# Strategy 2: For structured wastewater records
def chunk_structured_document(document):
    fields = extract_fields(document.page_content)
    source = document.metadata.get("source", "unknown")
    file_type = document.metadata.get("file_type", "unknown")

    company = fields.get("Company", "Unknown")
    location = fields.get("Location", "Unknown")
    industry = fields.get("Industry", "Unknown")

    # Keep important context in each chunk so that a
    # retrieved chunk can still identify its company.
    common_metadata = {
        **document.metadata,
        "company": company,
        "location": location,
        "industry": industry,
        "chunking_strategy": "field_aware"
    }

    groups = [
        (
            "company_information",
            [
                "Company",
                "Location",
                "Industry"
            ]
        ),
        (
            "treatment_information",
            [
                "Wastewater Generation",
                "Treatment Plant",
                "Treatment Technology",
                "Treatment Capacity"
            ]
        ),
        (
            "treatment_process",
            [
                "Primary Treatment",
                "Secondary Treatment",
                "Tertiary Treatment",
                "Treated Water Reuse",
                "Disposal Method"
            ]
        ),
        (
            "source_evidence",
            [
                "Source",
                "Page Number",
                "Evidence"
            ]
        )
    ]

    chunks = []

    for chunk_type, field_names in groups:
        lines = [f"Company: {company}"]

        for field_name in field_names:
            value = fields.get(field_name)

            if value:
                lines.append(f"{field_name}: {value}")

        # Do not create an empty or context-only chunk.
        if len(lines) == 1:
            continue

        chunks.append(
            Document(
                page_content="\n".join(lines),
                metadata={
                    **common_metadata,
                    "chunk_type": chunk_type
                }
            )
        )

    return chunks


# Strategy 3: For paragraphs and ordinary prose
def chunk_paragraph_document(document):
    chunks = text_splitter.split_documents([document])

    for chunk in chunks:
        chunk.metadata["chunking_strategy"] = "recursive_text"
        chunk.metadata["chunk_type"] = "paragraph"

    return chunks


# Dispatcher: choose a strategy for each document
def process_document(document):
    file_type = document.metadata.get("file_type", "").lower()

    if file_type in {"txt", "text"}:
        if is_structured_text(document.page_content):
            return chunk_structured_document(document)

        return chunk_paragraph_document(document)

    # Other file types will get their own extraction
    # and chunking strategies as we add their loaders.
    raise ValueError(
        f"No chunking strategy configured for file type: {file_type}"
    )


if __name__ == "__main__":
    from load_documents import load_text_documents

    documents = load_text_documents()
    all_chunks = []

    for document in documents:
        all_chunks.extend(process_document(document))

    print(f"Number of input documents: {len(documents)}")
    print(f"Number of chunks: {len(all_chunks)}")

    for index, chunk in enumerate(all_chunks, start=1):
        print("\n" + "=" * 45)
        print(f"Chunk {index}")
        print("Company:", chunk.metadata.get("company", "Not identified"))
        print("Strategy:", chunk.metadata["chunking_strategy"])
        print("Type:", chunk.metadata["chunk_type"])
        print("Source:", chunk.metadata.get("source"))
        print("-" * 45)
        print(chunk.page_content)
