"""Supabase pgvector and Storage adapters used by the hosted application.

The small collection-like facade intentionally mirrors the Chroma methods used
by the existing API, so structured wastewater handlers can keep their current
query logic while the hosted records live in Postgres.
"""

from __future__ import annotations

import os
from pathlib import Path
import psycopg
import requests
from langchain_core.documents import Document
from psycopg.types.json import Jsonb


VECTOR_DIMENSIONS = 384  # Matches the existing Supabase pgvector column.
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
GEMINI_EMBEDDING_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{DEFAULT_EMBEDDING_MODEL}:batchEmbedContents"
)
_SCHEMA_READY = False


def _database_url() -> str:
    value = os.getenv("DATABASE_URL", "").strip()
    if not value:
        raise RuntimeError("DATABASE_URL is required for the cloud vector database.")
    return value


def _supabase_config() -> tuple[str, str, str]:
    url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    key = os.getenv("SUPABASE_SECRET_KEY", "").strip()
    bucket = os.getenv("SUPABASE_BUCKET", "rag-documents").strip()
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SECRET_KEY are required.")
    return url, key, bucket


def ensure_schema() -> None:
    """Create the app tables once; pgvector must be enabled in Supabase first."""
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    with psycopg.connect(_database_url(), connect_timeout=15) as connection:
        with connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(
                f"""CREATE TABLE IF NOT EXISTS rag_documents (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector({VECTOR_DIMENSIONS}) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )"""
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS rag_documents_metadata_gin "
                "ON rag_documents USING GIN (metadata)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS rag_documents_embedding_hnsw "
                "ON rag_documents USING hnsw (embedding vector_cosine_ops)"
            )
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS rag_uploaded_files (
                    stored_filename TEXT PRIMARY KEY,
                    original_filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    size_bytes BIGINT NOT NULL,
                    sha256 TEXT NOT NULL UNIQUE,
                    indexed_records INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )"""
            )
    _SCHEMA_READY = True


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in vector) + "]"


def _embedding_values(payload: dict) -> list[float]:
    values = payload.get("values") if isinstance(payload, dict) else None
    if not isinstance(values, list) or len(values) != VECTOR_DIMENSIONS:
        actual = len(values) if isinstance(values, list) else "missing"
        raise RuntimeError(
            f"Gemini embedding returned {actual} values; expected "
            f"{VECTOR_DIMENSIONS}."
        )
    return [float(value) for value in values]


def embed_texts(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """Request low-dimensional remote vectors from Gemini Embedding 2."""
    token = os.getenv("GEMINI_API_KEY", "").strip()
    if not token:
        raise RuntimeError("GEMINI_API_KEY is required for hosted embeddings on Render.")
    vectors: list[list[float]] = []
    # Small batches limit both request size and memory use on Render's free tier.
    for start in range(0, len(texts), 12):
        batch = texts[start:start + 12]
        if input_type == "query":
            prepared = [f"task: question answering | query: {text}" for text in batch]
        else:
            prepared = [f"title: none | text: {text}" for text in batch]
        response = requests.post(
            GEMINI_EMBEDDING_URL,
            headers={"x-goog-api-key": token},
            json={
                "requests": [
                    {
                        "model": f"models/{DEFAULT_EMBEDDING_MODEL}",
                        "content": {"parts": [{"text": text}]},
                        "outputDimensionality": VECTOR_DIMENSIONS,
                    }
                    for text in prepared
                ]
            },
            timeout=90,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Gemini embedding request failed ({response.status_code}): "
                f"{response.text[:500]}"
            )
        payload = response.json()
        embeddings = payload.get("embeddings") if isinstance(payload, dict) else None
        if not isinstance(embeddings, list) or len(embeddings) != len(batch):
            raise RuntimeError("Gemini returned an unexpected embedding response.")
        vectors.extend(_embedding_values(item) for item in embeddings)
    return vectors


def _filter_sql(where) -> tuple[str, list]:
    if not where:
        return "", []
    if "$and" in where:
        children = [_filter_sql(item) for item in where["$and"]]
        clauses = [sql for sql, _ in children if sql]
        params = [param for _, values in children for param in values]
        return (" AND ".join(f"({part})" for part in clauses), params) if clauses else ("", [])
    clauses, params = [], []
    for key, expected in where.items():
        if isinstance(expected, dict):
            if "$in" in expected:
                values = expected["$in"]
                if not values:
                    clauses.append("FALSE")
                else:
                    clauses.append("metadata ->> %s = ANY(%s)")
                    params.extend([key, [str(value) for value in values]])
            elif "$ne" in expected:
                clauses.append("COALESCE(metadata ->> %s, '') <> %s")
                params.extend([key, str(expected["$ne"])])
            else:
                raise ValueError(f"Unsupported metadata filter: {expected}")
        else:
            clauses.append("metadata ->> %s = %s")
            params.extend([key, str(expected)])
    return " AND ".join(clauses), params


class CloudCollection:
    def __init__(self, store: "CloudVectorStore"):
        self.store = store

    def get(self, where=None, include=None, limit=None, offset=0):
        ensure_schema()
        clause, params = _filter_sql(where)
        fields = "id, content, metadata"
        sql = f"SELECT {fields} FROM rag_documents"
        if clause:
            sql += f" WHERE {clause}"
        sql += " ORDER BY id"
        if limit is not None:
            sql += " LIMIT %s OFFSET %s"
            params.extend([int(limit), int(offset)])
        elif offset:
            sql += " OFFSET %s"
            params.append(int(offset))
        with psycopg.connect(_database_url(), connect_timeout=15) as connection:
            rows = connection.execute(sql, params).fetchall()
        return {
            "ids": [row[0] for row in rows],
            "documents": [row[1] for row in rows],
            "metadatas": [row[2] for row in rows],
        }

    def delete(self, ids=None, where=None):
        ensure_schema()
        if ids:
            with psycopg.connect(_database_url(), connect_timeout=15) as connection:
                connection.execute("DELETE FROM rag_documents WHERE id = ANY(%s)", [ids])
            return
        clause, params = _filter_sql(where)
        if not clause:
            raise ValueError("Refusing to delete all cloud vectors without a filter.")
        with psycopg.connect(_database_url(), connect_timeout=15) as connection:
            connection.execute(f"DELETE FROM rag_documents WHERE {clause}", params)

    def count(self):
        ensure_schema()
        with psycopg.connect(_database_url(), connect_timeout=15) as connection:
            return connection.execute("SELECT count(*) FROM rag_documents").fetchone()[0]


class CloudVectorStore:
    def __init__(self):
        ensure_schema()
        self._collection = CloudCollection(self)

    def add_documents(self, documents, ids):
        if not documents:
            return
        vectors = embed_texts(
            [document.page_content for document in documents], input_type="document"
        )
        rows = [
            (document_id, document.page_content, Jsonb(document.metadata), _vector_literal(vector))
            for document_id, document, vector in zip(ids, documents, vectors)
        ]
        with psycopg.connect(_database_url(), connect_timeout=15) as connection:
            connection.executemany(
                """INSERT INTO rag_documents (id, content, metadata, embedding)
                   VALUES (%s, %s, %s, %s::vector)
                   ON CONFLICT (id) DO UPDATE SET content=EXCLUDED.content,
                     metadata=EXCLUDED.metadata, embedding=EXCLUDED.embedding""",
                rows,
            )

    def similarity_search_with_score(self, query, k=5, filter=None):
        vector = _vector_literal(embed_texts([query], input_type="query")[0])
        clause, params = _filter_sql(filter)
        sql = "SELECT content, metadata, embedding <=> %s::vector AS distance FROM rag_documents"
        params.insert(0, vector)
        if clause:
            sql += f" WHERE {clause}"
        sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([vector, int(k)])
        with psycopg.connect(_database_url(), connect_timeout=15) as connection:
            rows = connection.execute(sql, params).fetchall()
        return [(Document(page_content=row[0], metadata=row[1] or {}), float(row[2])) for row in rows]

    def similarity_search(self, query, k=5, filter=None):
        return [document for document, _ in self.similarity_search_with_score(query, k=k, filter=filter)]


def cloud_mode_enabled() -> bool:
    names = ("DATABASE_URL", "SUPABASE_URL", "SUPABASE_SECRET_KEY")
    configured = [bool(os.getenv(name, "").strip()) for name in names]
    if any(configured) and not all(configured):
        missing = [name for name, is_set in zip(names, configured) if not is_set]
        raise RuntimeError("Cloud storage configuration is incomplete; missing: " + ", ".join(missing))
    return all(configured)


def upload_file_object(path: str | Path, stored_filename: str, content_type: str) -> None:
    url, key, bucket = _supabase_config()
    object_url = f"{url}/storage/v1/object/{quote(bucket, safe='')}/{quote(stored_filename, safe='')}"
    with Path(path).open("rb") as source:
        response = requests.put(
            object_url,
            headers={
                "Authorization": f"Bearer {key}",
                "apikey": key,
                "Content-Type": content_type or "application/octet-stream",
                "x-upsert": "true",
            },
            data=source,
            timeout=120,
        )
    if response.status_code >= 400:
        raise RuntimeError(f"Supabase Storage upload failed ({response.status_code}): {response.text[:400]}")


def download_file_object(stored_filename: str, destination: str | Path) -> Path:
    url, key, bucket = _supabase_config()
    object_url = f"{url}/storage/v1/object/{quote(bucket, safe='')}/{quote(stored_filename, safe='')}"
    response = requests.get(
        object_url,
        headers={"Authorization": f"Bearer {key}", "apikey": key},
        timeout=120,
    )
    if response.status_code >= 400:
        raise FileNotFoundError(f"Supabase Storage download failed ({response.status_code}).")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(response.content)
    return target


def delete_file_object(stored_filename: str) -> None:
    url, key, bucket = _supabase_config()
    response = requests.delete(
        f"{url}/storage/v1/object/{quote(bucket, safe='')}",
        headers={"Authorization": f"Bearer {key}", "apikey": key},
        json={"prefixes": [stored_filename]},
        timeout=30,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Supabase Storage delete failed ({response.status_code}): {response.text[:400]}")


def register_file(*, stored_filename, original_filename, file_type, size_bytes, sha256, indexed_records=0):
    ensure_schema()
    with psycopg.connect(_database_url(), connect_timeout=15) as connection:
        connection.execute(
            """INSERT INTO rag_uploaded_files
               (stored_filename, original_filename, file_type, size_bytes, sha256, indexed_records)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON CONFLICT (stored_filename) DO UPDATE SET original_filename=EXCLUDED.original_filename,
                 file_type=EXCLUDED.file_type, size_bytes=EXCLUDED.size_bytes,
                 sha256=EXCLUDED.sha256, indexed_records=EXCLUDED.indexed_records""",
            (stored_filename, original_filename, file_type, size_bytes, sha256, indexed_records),
        )


def find_file_by_hash(sha256: str):
    ensure_schema()
    with psycopg.connect(_database_url(), connect_timeout=15) as connection:
        row = connection.execute(
            "SELECT stored_filename, original_filename, file_type, size_bytes, sha256, indexed_records "
            "FROM rag_uploaded_files WHERE sha256=%s", [sha256],
        ).fetchone()
    return _file_row(row) if row else None


def get_file(stored_filename: str):
    ensure_schema()
    with psycopg.connect(_database_url(), connect_timeout=15) as connection:
        row = connection.execute(
            "SELECT stored_filename, original_filename, file_type, size_bytes, sha256, indexed_records "
            "FROM rag_uploaded_files WHERE stored_filename=%s", [stored_filename],
        ).fetchone()
    return _file_row(row) if row else None


def list_files():
    ensure_schema()
    with psycopg.connect(_database_url(), connect_timeout=15) as connection:
        rows = connection.execute(
            "SELECT stored_filename, original_filename, file_type, size_bytes, sha256, indexed_records "
            "FROM rag_uploaded_files ORDER BY original_filename"
        ).fetchall()
    return [_file_row(row) for row in rows]


def delete_file_record(stored_filename: str):
    ensure_schema()
    with psycopg.connect(_database_url(), connect_timeout=15) as connection:
        connection.execute("DELETE FROM rag_uploaded_files WHERE stored_filename=%s", [stored_filename])


def _file_row(row):
    keys = ("stored_filename", "original_filename", "file_type", "size_bytes", "sha256", "indexed_records")
    return dict(zip(keys, row))


def original_filename_for(source: str) -> str | None:
    filename = Path(source or "").name
    record = get_file(filename)
    return record["original_filename"] if record else None
