"""General retrieval over every indexed content type.

The old generic search filtered on ``content_type == "table_row"``, which made
TXT, PDF narrative, DOCX and CSV content unreachable. This module searches all
content types and fuses two signals:

* semantic similarity from Chroma (good for paraphrased questions), and
* BM25 keyword ranking (good for unique names, codes and numbers that a small
  embedding model blurs together).

Legacy STP ``table_row`` records are excluded from the keyword corpus because
the structured STP handlers already own them; they can still surface through
the semantic leg when nothing better exists.

Pure helpers (tokenise, bm25_rank, fuse_hits) have no Chroma dependency.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from src.config import UPLOAD_METADATA_DIR

# all-MiniLM-L6-v2 vectors are unit length, so L2 distance is in [0, 2].
# 1.0 corresponds to a cosine similarity of 0.5.
MAX_SEMANTIC_DISTANCE = 1.0

STOPWORDS = frozenset("""
a an the and or but of in on at to for from by with about as is are was were be been being
this that these those it its what which who whom whose when where why how do does did has have
had can could should would will shall may might must not no yes tell me please give show list
find any all some there their they them he she his her you your we our i my mine than then so
if into out up down over under between per also more most less least very much many such
document documents file files uploaded upload report information info details detail
""".split())


def tokenize(text: str) -> list[str]:
    tokens = []
    for token in re.findall(r"[a-z0-9][a-z0-9._/\-]*[a-z0-9]|[a-z0-9]", (text or "").lower()):
        token = token.strip("._-/")
        if not token or token in STOPWORDS:
            continue
        # very light plural stripping so "tanks" matches "tank"
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return tokens


def bm25_rank(query: str, corpus: list[dict], k1: float = 1.5, b: float = 0.75) -> list[dict]:
    """Rank ``corpus`` items (dicts with a 'content' key) against ``query``.

    Returns items with a positive score, best first. Each result also gets a
    ``coverage`` value: the fraction of distinct query terms found in it.
    """
    query_terms = list(dict.fromkeys(tokenize(query)))
    if not query_terms or not corpus:
        return []

    docs = [Counter(tokenize(item["content"])) for item in corpus]
    lengths = [sum(doc.values()) or 1 for doc in docs]
    average = sum(lengths) / len(lengths)
    total = len(corpus)

    document_frequency = {
        term: sum(1 for doc in docs if term in doc) for term in query_terms
    }

    ranked = []
    for item, doc, length in zip(corpus, docs, lengths):
        score = 0.0
        found = 0
        for term in query_terms:
            frequency = doc.get(term, 0)
            if not frequency:
                continue
            found += 1
            idf = math.log(1 + (total - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            score += idf * (frequency * (k1 + 1)) / (frequency + k1 * (1 - b + b * length / average))
        if score > 0:
            ranked.append({**item, "bm25": score, "coverage": found / len(query_terms)})

    ranked.sort(key=lambda hit: hit["bm25"], reverse=True)
    return ranked


def _identity(hit: dict) -> tuple:
    return (hit.get("source"), hit.get("content_type"), hit.get("page"),
            hit.get("sheet"), hit.get("table_number"), hit.get("row_number"),
            hit.get("chunk_index"), hit.get("content"))


def fuse_hits(
    semantic: list[dict],
    lexical: list[dict],
    k: int = 5,
    max_distance: float = MAX_SEMANTIC_DISTANCE,
) -> list[dict]:
    """Reciprocal-rank fusion with a relevance gate.

    A hit survives if it shares at least one non-stopword term with the
    question (lexical hit) OR its semantic distance is within ``max_distance``.
    Everything else is noise and is dropped, so an unrelated question yields
    no sources instead of arbitrary nearest neighbours.
    """
    merged: dict[tuple, dict] = {}

    for rank, hit in enumerate(semantic):
        entry = merged.setdefault(_identity(hit), {**hit, "rrf": 0.0, "coverage": hit.get("coverage", 0.0)})
        entry["distance"] = hit.get("distance")
        entry["rrf"] += 1.0 / (60 + rank)

    for rank, hit in enumerate(lexical):
        entry = merged.setdefault(_identity(hit), {**hit, "rrf": 0.0})
        entry["coverage"] = max(entry.get("coverage", 0.0), hit.get("coverage", 0.0))
        entry["bm25"] = hit.get("bm25")
        entry["rrf"] += 1.0 / (60 + rank)

    kept = [
        hit for hit in merged.values()
        if hit.get("bm25") or (hit.get("distance") is not None and hit["distance"] <= max_distance)
    ]
    kept.sort(key=lambda hit: hit["rrf"], reverse=True)
    return kept[:k]


# --------------------------------------------------------------------------
# Chroma-backed search
# --------------------------------------------------------------------------

def _to_hit(content, metadata, distance=None) -> dict:
    metadata = metadata or {}
    return {
        "content": content or "",
        "source": metadata.get("source", "Unknown"),
        "file_name": metadata.get("file_name"),
        "file_type": metadata.get("file_type"),
        "content_type": metadata.get("content_type"),
        "page": metadata.get("page"),
        "sheet": metadata.get("sheet"),
        "table_number": metadata.get("table_number"),
        "row_number": metadata.get("row_number"),
        "chunk_index": metadata.get("chunk_index"),
        "distance": distance,
    }


def _where(file_types, sources, extra=None):
    clauses = []
    if file_types:
        clauses.append({"file_type": {"$in": list(file_types)}})
    if sources:
        clauses.append({"source": {"$in": list(sources)}})
    if extra:
        clauses.append(extra)
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _lexical_corpus(collection, file_types, sources) -> list[dict]:
    where = _where(file_types, sources, {"content_type": {"$ne": "table_row"}})
    corpus, offset, batch = [], 0, 1000

    while True:
        page = collection.get(where=where, limit=batch, offset=offset,
                              include=["documents", "metadatas"])
        documents = page.get("documents") or []
        metadatas = page.get("metadatas") or []
        corpus.extend(_to_hit(c, m) for c, m in zip(documents, metadatas))
        offset += len(documents)
        if len(documents) < batch:
            break

    return corpus


def general_search(
    question: str,
    k: int = 5,
    file_types: list[str] | None = None,
    sources: list[str] | None = None,
    vectorstore=None,
) -> list[dict]:
    """Search all content types; optionally restrict by file type or source."""
    if vectorstore is None:
        from src.vectorstore.search_index import create_vectorstore
        vectorstore = create_vectorstore()

    collection = vectorstore._collection
    where = _where(file_types, sources)

    semantic = []
    try:
        results = vectorstore.similarity_search_with_score(question, k=k * 3, filter=where)
        for document, distance in results:
            semantic.append(_to_hit(document.page_content, document.metadata, float(distance)))
    except Exception:
        # An empty collection or a bad filter must not break keyword search.
        semantic = []

    lexical = bm25_rank(question, _lexical_corpus(collection, file_types, sources))[: k * 3]

    return fuse_hits(semantic, lexical, k=k)


def display_name(source: str) -> str:
    """Original upload filename if a sidecar exists, else the file name."""
    try:
        from src.vectorstore.cloud_store import cloud_mode_enabled, original_filename_for
        if cloud_mode_enabled():
            return original_filename_for(source) or Path(source or "").name or "Unknown"
    except Exception:
        pass
    path = Path(source or "")
    sidecar = UPLOAD_METADATA_DIR / f"{path.name}.json"
    try:
        if sidecar.is_file():
            return json.loads(sidecar.read_text()).get("original_filename", path.name)
    except (OSError, json.JSONDecodeError):
        pass
    return path.name or "Unknown"


def best_excerpt(question: str, content: str, max_chars: int = 600) -> str:
    """Pick the lines/sentences of ``content`` that best match the question.

    Used when the LLM is unavailable, so the user still gets the specific
    passage rather than an error or an unrelated summary.
    """
    terms = set(tokenize(question))
    pieces = [
        piece.strip()
        for piece in re.split(r"(?<=[.!?])\s+|\n+", content or "")
        if piece.strip()
    ]
    if not pieces:
        return ""

    scored = [
        (len(terms & set(tokenize(piece))), index, piece)
        for index, piece in enumerate(pieces)
    ]
    best = sorted(scored, key=lambda item: (-item[0], item[1]))[:3]
    best = [item for item in best if item[0] > 0] or best[:1]

    excerpt = "\n".join(piece for _, _, piece in sorted(best, key=lambda item: item[1]))
    return excerpt[:max_chars]
