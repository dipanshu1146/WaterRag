import hashlib
import json
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from src.config import UPLOAD_DIR, UPLOAD_METADATA_DIR as METADATA_DIR
from src.ingestion.pipeline import UnsupportedDocumentError
from src.vectorstore.search_index import (
    create_vectorstore,
    delete_source_records,
    index_uploaded_file,
    normalize_source,
)

router = APIRouter(prefix="/api/documents", tags=["Documents"])

# Legacy .xls is deliberately NOT supported (openpyxl cannot read it), and the
# frontend file picker matches this list.
ALLOWED_EXTENSIONS = {".pdf", ".xlsx", ".csv", ".docx", ".txt"}
MAX_FILE_SIZE = 25 * 1024 * 1024


def calculate_sha256(file_path: Path) -> str:
    """Calculate a file's SHA-256 hash without loading it all into memory."""
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _sidecar(stored_name: str) -> Path:
    return METADATA_DIR / f"{stored_name}.json"


def _read_sidecar(stored_name: str) -> dict:
    try:
        return json.loads(_sidecar(stored_name).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _find_duplicate(uploaded_hash: str, exclude: Path):
    """Return an existing stored file with identical bytes, if any.

    The hash saved in the sidecar is used when available so uploads do not
    re-read every stored file.
    """
    for existing in UPLOAD_DIR.iterdir():
        if not existing.is_file() or existing == exclude or existing.name.startswith("."):
            continue

        known_hash = _read_sidecar(existing.name).get("sha256")
        if (known_hash or calculate_sha256(existing)) == uploaded_hash:
            return existing

    return None


def _remove_file_and_sidecar(stored_name: str):
    (UPLOAD_DIR / stored_name).unlink(missing_ok=True)
    _sidecar(stored_name).unlink(missing_ok=True)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    original_name = Path(file.filename or "").name
    extension = Path(original_name).suffix.lower()

    if not original_name or extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload PDF, Excel (.xlsx), CSV, DOCX, or TXT.",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    stored_name = f"{uuid4().hex}{extension}"
    destination = UPLOAD_DIR / stored_name

    size = 0
    digest = hashlib.sha256()
    keep_saved_file = False

    try:
        # Save the upload while calculating its hash.
        with destination.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)

                if size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="File exceeds the 25 MB upload limit.",
                    )

                digest.update(chunk)
                output.write(chunk)

        uploaded_hash = digest.hexdigest()

        duplicate = _find_duplicate(uploaded_hash, destination)
        if duplicate is not None:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "This file has already been uploaded.",
                    "existing_filename": duplicate.name,
                },
            )

        _sidecar(stored_name).write_text(
            json.dumps(
                {
                    "original_filename": original_name,
                    "stored_filename": stored_name,
                    "file_type": extension.lstrip("."),
                    "size_bytes": size,
                    "sha256": uploaded_hash,
                },
                indent=2,
            )
        )
        keep_saved_file = True

        try:
            indexed_records = await run_in_threadpool(index_uploaded_file, str(destination))
        except UnsupportedDocumentError as exc:
            # The file has nothing we can index. Retrying the same bytes cannot
            # help, so do not keep it (and do not block a corrected re-upload).
            _remove_file_and_sidecar(stored_name)
            keep_saved_file = False
            raise HTTPException(
                status_code=422,
                detail={
                    "message": f"File was not indexed: {exc}",
                    "original_filename": original_name,
                    "status": "not_indexed",
                    "indexed_records": 0,
                },
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": (
                        "File was saved, but indexing failed. Use the Re-index "
                        "button in the document list to try again."
                    ),
                    "original_filename": original_name,
                    "stored_filename": stored_name,
                    "error": str(exc),
                },
            ) from exc

        # Defence in depth: never report success for zero records.
        if indexed_records <= 0:
            _remove_file_and_sidecar(stored_name)
            keep_saved_file = False
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "File was not indexed: no readable content was found.",
                    "original_filename": original_name,
                    "status": "not_indexed",
                    "indexed_records": 0,
                },
            )

        return {
            "message": f"File uploaded and indexed successfully ({indexed_records} records).",
            "original_filename": original_name,
            "stored_filename": stored_name,
            "file_type": extension.lstrip("."),
            "size_bytes": size,
            "sha256": uploaded_hash,
            "status": "indexed",
            "indexed_records": indexed_records,
        }

    except HTTPException:
        # Keep a completed file when indexing crashed (it can be re-indexed),
        # but clean up partial uploads and files rejected during validation.
        if not keep_saved_file:
            _remove_file_and_sidecar(stored_name)
        raise

    except OSError as exc:
        _remove_file_and_sidecar(stored_name)
        raise HTTPException(
            status_code=500,
            detail="Could not save or check the uploaded file.",
        ) from exc

    finally:
        await file.close()


@router.get("")
def list_uploaded_documents():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        collection = create_vectorstore()._collection
        indexed = collection.get(include=["metadatas"])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read indexed document metadata: {exc}",
        ) from exc

    # Count by canonical absolute path so relative and absolute sources agree.
    counts = {}
    for metadata in indexed.get("metadatas") or []:
        source_path = (metadata or {}).get("source")
        if source_path:
            key = normalize_source(source_path)
            counts[key] = counts.get(key, 0) + 1

    documents = []
    for path in sorted(UPLOAD_DIR.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.name.startswith("."):
            continue

        saved = _read_sidecar(path.name)
        record_count = counts.get(str(path.resolve()), 0)

        documents.append({
            "original_filename": saved.get("original_filename", path.name),
            "stored_filename": path.name,
            "size_bytes": path.stat().st_size,
            "indexed_records": record_count,
            "status": "indexed" if record_count > 0 else "saved, not indexed",
        })

    return {"documents": documents, "total": len(documents)}


def _validated_target(stored_filename: str) -> Path:
    if (
        Path(stored_filename).name != stored_filename
        or stored_filename.startswith(".")
    ):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    target = UPLOAD_DIR / stored_filename
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Document not found.")

    return target


@router.post("/{stored_filename}/reindex")
async def reindex_uploaded_document(stored_filename: str):
    """Re-run indexing for a saved file (e.g. after a failed attempt)."""
    target = _validated_target(stored_filename)

    try:
        indexed_records = await run_in_threadpool(index_uploaded_file, str(target))
    except UnsupportedDocumentError as exc:
        raise HTTPException(
            status_code=422,
            detail={"message": f"File was not indexed: {exc}", "status": "not_indexed"},
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"message": "Re-indexing failed.", "error": str(exc)},
        ) from exc

    return {
        "message": f"File re-indexed successfully ({indexed_records} records).",
        "stored_filename": stored_filename,
        "status": "indexed",
        "indexed_records": indexed_records,
    }


@router.delete("/{stored_filename}")
def delete_uploaded_document(stored_filename: str):
    target = _validated_target(stored_filename)

    try:
        # Matches records indexed with absolute OR relative source paths.
        deleted = delete_source_records(target)

        target.unlink()
        _sidecar(stored_filename).unlink(missing_ok=True)

        return {
            "message": "Document deleted.",
            "stored_filename": stored_filename,
            "deleted_indexed_records": deleted,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not delete document: {exc}",
        ) from exc
