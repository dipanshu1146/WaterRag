"""Shared fixtures.

The tests run fully offline: no Ollama, no model download, no on-disk Chroma.
``FakeVectorStore`` implements just the slice of the Chroma / LangChain API
the application uses (get / delete / add_documents /
similarity_search_with_score with metadata filters), so the REAL upload,
ingestion, retrieval and /api/ask code is exercised end to end.
"""

import math
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _matches(metadata, where):
    if not where:
        return True
    for key, condition in where.items():
        if key == "$and":
            if not all(_matches(metadata, clause) for clause in condition):
                return False
        elif key == "$or":
            if not any(_matches(metadata, clause) for clause in condition):
                return False
        elif isinstance(condition, dict):
            value = metadata.get(key)
            for op, operand in condition.items():
                if op == "$in" and value not in operand:
                    return False
                if op == "$nin" and value in operand:
                    return False
                if op == "$ne" and value == operand:
                    return False
                if op == "$eq" and value != operand:
                    return False
        elif metadata.get(key) != condition:
            return False
    return True


def _words(text):
    return re.findall(r"[a-z0-9]+", (text or "").lower())


class _Collection:
    def __init__(self, store):
        self.store = store

    def get(self, ids=None, where=None, limit=None, offset=0, include=None):
        rows = [
            (i, row) for i, row in self.store.rows.items()
            if (ids is None or i in ids) and _matches(row["metadata"], where)
        ]
        rows = rows[offset:] if limit is None else rows[offset:offset + limit]
        return {
            "ids": [i for i, _ in rows],
            "documents": [row["content"] for _, row in rows],
            "metadatas": [row["metadata"] for _, row in rows],
        }

    def delete(self, ids=None, where=None):
        for i in list(ids or []):
            self.store.rows.pop(i, None)


class FakeVectorStore:
    """Dict-backed stand-in for langchain_chroma.Chroma."""

    def __init__(self):
        self.rows = {}
        self._collection = _Collection(self)

    def add_documents(self, documents, ids):
        for doc, doc_id in zip(documents, ids):
            for key, value in doc.metadata.items():
                # Real Chroma rejects non-scalar metadata values.
                assert isinstance(value, (str, int, float, bool)), (key, value)
            self.rows[doc_id] = {"content": doc.page_content, "metadata": dict(doc.metadata)}

    def similarity_search(self, query, k=4, filter=None):
        return [doc for doc, _ in self.similarity_search_with_score(query, k=k, filter=filter)]

    def similarity_search_with_score(self, query, k=4, filter=None):
        from langchain_core.documents import Document

        query_words = set(_words(query))
        scored = []
        for row in self.rows.values():
            if not _matches(row["metadata"], filter):
                continue
            words = set(_words(row["content"]))
            overlap = len(query_words & words)
            cosine = overlap / math.sqrt(max(len(query_words), 1) * max(len(words), 1))
            scored.append((1.0 - cosine, row))
        scored.sort(key=lambda item: item[0])
        return [
            (Document(page_content=row["content"], metadata=dict(row["metadata"])), 2 * distance)
            for distance, row in scored[:k]
        ]


class _FailingLLM:
    """Forces the extractive (no-LLM) answer path so tests are deterministic."""

    def invoke(self, prompt):
        raise RuntimeError("LLM disabled in tests")


@pytest.fixture()
def store(monkeypatch, tmp_path):
    fake = FakeVectorStore()

    import src.api.routes.ask as ask
    import src.api.routes.upload as upload
    import src.vectorstore.general_search as general_search
    import src.vectorstore.search_index as search_index

    uploads = tmp_path / "uploads"
    metadata = uploads / ".metadata"

    for module in (search_index, upload, ask):
        monkeypatch.setattr(module, "create_vectorstore", lambda: fake)
    monkeypatch.setattr(upload, "UPLOAD_DIR", uploads)
    monkeypatch.setattr(upload, "METADATA_DIR", metadata)
    monkeypatch.setattr(general_search, "UPLOAD_METADATA_DIR", metadata)
    monkeypatch.setattr(ask, "llm", _FailingLLM())
    monkeypatch.setattr(ask, "load_all_table_rows", lambda: [])
    # Bulk-indexed relative sources resolve against the project root.
    return fake


@pytest.fixture()
def client(store):
    from fastapi.testclient import TestClient

    from src.api.main import app

    return TestClient(app)
