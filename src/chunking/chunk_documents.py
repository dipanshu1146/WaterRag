
import re


def split_text(text: str, chunk_size: int = 1200, overlap: int = 200):
    """Split narrative text while retaining a small overlap."""

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0.")

    # Prefer paragraph and sentence boundaries over arbitrary cuts.
    paragraphs = re.split(r"\n\s*\n", text.strip())
    chunks = []
    current = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # Split unusually long paragraphs into smaller pieces.
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current)
                current = ""

            start = 0
            while start < len(paragraph):
                end = min(start + chunk_size, len(paragraph))

                if end < len(paragraph):
                    boundary = paragraph.rfind(" ", start, end)
                    if boundary > start:
                        end = boundary

                piece = paragraph[start:end].strip()
                if piece:
                    chunks.append(piece)

                if end >= len(paragraph):
                    break

                start = max(end - overlap, start + 1)

            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)

            # Carry a short overlap from the preceding chunk.
            current = current[-overlap:] if overlap else ""
            current = f"{current}\n\n{paragraph}".strip()

            if len(current) > chunk_size:
                chunks.append(current[:chunk_size])
                current = current[chunk_size - overlap:] if overlap else ""

    if current.strip():
        chunks.append(current.strip())

    return chunks


def split_table(text: str, rows_per_chunk: int = 20):
    """Keep a table's header with each group of data rows."""

    if rows_per_chunk < 1:
        raise ValueError("rows_per_chunk must be at least 1.")

    rows = [line.strip() for line in text.splitlines() if line.strip()]

    if not rows:
        return []

    header = rows[0]
    data_rows = rows[1:]
    chunks = []

    for start in range(0, len(data_rows), rows_per_chunk):
        batch = data_rows[start:start + rows_per_chunk]
        chunks.append("\n".join([header, *batch]))

    # A one-row table still needs to be retained.
    if not data_rows:
        chunks.append(header)

    return chunks


def chunk_record(record: dict, chunk_size: int = 1200):
    """Chunk one extracted record and preserve its source metadata."""

    content = record.get("content", "").strip()
    if not content:
        return []

    if record.get("content_type") == "table":
        pieces = split_table(content, rows_per_chunk=20)
    else:
        pieces = split_text(content, chunk_size=chunk_size, overlap=200)

    result = []

    for index, piece in enumerate(pieces):
        result.append({
            "content": piece,
            "metadata": {
                key: value
                for key, value in record.items()
                if key != "content"
            } | {
                "chunk_index": index,
                "chunk_type": record.get("content_type", "text"),
            },
        })

    return result


def chunk_records(records: list[dict]):
    chunks = []

    for record in records:
        chunks.extend(chunk_record(record))

    return chunks
