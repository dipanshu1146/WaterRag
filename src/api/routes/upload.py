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
from src.vectorstore.cloud_store import (
    cloud_mode_enabled,
    delete_file_object,
    delete_file_record,
    download_file_object,
    find_file_by_hash,
    get_file,
    list_files,
    register_file,
    upload_file_object,
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
    cloud_mode = cloud_mode_enabled()

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

        duplicate = (
            find_file_by_hash(uploaded_hash)
            if cloud_mode
            else _find_duplicate(uploaded_hash, destination)
        )
        if duplicate is not None:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "This file has already been uploaded.",
                    "existing_filename": (
                        duplicate["stored_filename"] if cloud_mode else duplicate.name
                    ),
                },
            )

        if cloud_mode:
            upload_file_object(destination, stored_name, file.content_type or "application/octet-stream")
            register_file(
                stored_filename=stored_name,
                original_filename=original_name,
                file_type=extension.lstrip("."),
                size_bytes=size,
                sha256=uploaded_hash,
                indexed_records=0,
            )
        else:
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
            if cloud_mode:
                delete_file_object(stored_name)
                delete_file_record(stored_name)
            else:
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

        if cloud_mode:
            register_file(
                stored_filename=stored_name,
                original_filename=original_name,
                file_type=extension.lstrip("."),
                size_bytes=size,
                sha256=uploaded_hash,
                indexed_records=indexed_records,
            )

        # Defence in depth: never report success for zero records.
        if indexed_records <= 0:
            if cloud_mode:
                delete_file_object(stored_name)
                delete_file_record(stored_name)
            else:
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
        if cloud_mode:
            destination.unlink(missing_ok=True)
        await file.close()


@router.get("")
def list_uploaded_documents():
    if cloud_mode_enabled():
        try:
            documents = [
                {
                    "original_filename": item["original_filename"],
                    "stored_filename": item["stored_filename"],
                    "size_bytes": item["size_bytes"],
                    "indexed_records": item["indexed_records"],
                    "status": "indexed" if item["indexed_records"] > 0 else "saved, not indexed",
                }
                for item in list_files()
            ]
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not read cloud document metadata: {exc}") from exc
        return {"documents": documents, "total": len(documents)}

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
    if cloud_mode_enabled():
        record = get_file(stored_filename)
        if record is None:
            raise HTTPException(status_code=404, detail="Document not found.")
        target = UPLOAD_DIR / stored_filename
        try:
            download_file_object(stored_filename, target)
            indexed_records = await run_in_threadpool(index_uploaded_file, str(target))
            record["indexed_records"] = indexed_records
            register_file(**record)
        except Exception as exc:
            raise HTTPException(status_code=500, detail={"message": "Re-indexing failed.", "error": str(exc)}) from exc
        finally:
            target.unlink(missing_ok=True)
        return {
            "message": f"File re-indexed successfully ({indexed_records} records).",
            "stored_filename": stored_filename,
            "status": "indexed",
            "indexed_records": indexed_records,
        }

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
    if cloud_mode_enabled():
        record = get_file(stored_filename)
        if record is None:
            raise HTTPException(status_code=404, detail="Document not found.")
        try:
            deleted = delete_source_records(stored_filename)
            delete_file_object(stored_filename)
            delete_file_record(stored_filename)
            return {
                "message": "Document deleted.",
                "stored_filename": stored_filename,
                "deleted_indexed_records": deleted,
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not delete document: {exc}") from exc

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
