# Code-audit fixes: what changed

## Fixed
| Audit finding | Fix | Where |
|---|---|---|
| **Critical:** malformed regexes → HTTP 500 on STP-ID questions | Rewrote the capacity and treatment-technology patterns | `src/api/routes/ask.py` |
| **High:** generic search excluded text / CSV / DOCX | New `general_search()` searches *all* content types: semantic + BM25 keyword fusion, with a relevance gate so unrelated questions return no sources | `src/vectorstore/general_search.py`, wired into `ask.py` |
| **High:** uploads bypassed chunking | One pipeline: extract → format-aware chunk → normalised metadata → index. TXT/PDF/DOCX text is split (~800 chars); CSV/DOCX/XLSX tables become one labelled record per row + a table summary | `src/ingestion/pipeline.py`, `search_index.index_uploaded_file` |
| **High:** "first TXT file / all workbooks" handler | Explicit TXT/CSV/DOCX questions now use relevance retrieval restricted to that file type. Whole-workbook summaries keep the old summary view | `ask.py::answer_uploaded_file_question` |
| **High:** Excel parser tied to one layout | Haryana parser is used only if the STP layout is detected; otherwise a generic header-detecting extractor runs (`file_type="spreadsheet"`, kept separate from STP handlers) | `pipeline.py` |
| **High:** zero records reported as success | `UnsupportedDocumentError` → HTTP 422, file removed so a corrected upload isn't blocked as a duplicate; frontend also refuses to show success for 0 records | `pipeline.py`, `upload.py`, `App.jsx` |
| Scanned PDFs | OCR fallback (pytesseract + Tesseract) if installed; otherwise a clear error | `pipeline.py` |
| **Medium:** no retry | `POST /api/documents/{stored}/reindex` (idempotent) + Re-index button | `upload.py`, `App.jsx` |
| **Medium:** `.xls` mismatch | Removed from the file picker (backend never supported it) | `App.jsx` |
| **Medium:** delete misses relative paths | `delete_source_records()` matches absolute + relative spellings; list view normalises paths against the project root | `search_index.py`, `upload.py` |
| **Medium:** two Chroma paths / competing pipelines | `src/config.py` is the single source of truth (`data/chroma_db`, matching what the app already used); live uploads use only `pipeline.py` | `config.py` + vectorstore modules |
| **Lower:** TXT decoding | Tolerant decoding (UTF-8/UTF-16 BOM/cp1252/replace) | `pipeline.py`, `load_documents.py` |

## Also found and fixed (not in the audit)
* `requirements.txt` was missing `fastapi`, `python-multipart` (uploads cannot work without it), `pdfplumber`, `openpyxl`, `python-docx`, `pypdf`, `langchain-huggingface`, `sentence-transformers`.
* Frontend results table was hard-wired to STP columns (blank rows for general answers); icon always said "PDF".
* Facility-profile template answer fired for any "of <name>" question; now only for wastewater/treatment/reuse questions.
* CORS now also allows `127.0.0.1:5173`.

## Behaviour changes to be aware of
* Questions containing STP-database words (`stp`, `capacity`, `compliance`, `how many`, ...) still go to the structured STP path first; general retrieval is the fallback when it finds nothing.
* Non-structured questions now try general retrieval *before* the old PDF table-row semantic fallback.
* A generic `.xlsx` is `file_type="spreadsheet"`, so the STP handlers (which query `file_type="xlsx"`) never mis-parse it.

## Not changed
* `load_excel_documents` internals (incl. its `report_value` quirk), `build_index.py`, `src/ingestion/chunk_documents.py`, `src/chunking/*`: left in place, no longer used by uploads. Delete once you confirm.
* `.xls` reading, and the `*.bak` / `*.backup*` files (excluded from this zip).

## Running the tests
    pip install -r requirements.txt reportlab
    pytest -q
No Ollama, model download or on-disk Chroma needed (an in-memory fake store is used).
