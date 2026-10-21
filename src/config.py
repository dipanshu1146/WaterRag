
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Input data directories
DATA_DIR = BASE_DIR / "data"

PDF_DIR = DATA_DIR / "pdf"
DOCX_DIR = DATA_DIR / "docx"
EXCEL_DIR = DATA_DIR / "excel"
CSV_DIR = DATA_DIR / "csv"
TXT_DIR = DATA_DIR / "txt"
IMAGE_DIR = DATA_DIR / "images"
WEB_DIR = DATA_DIR / "web"

# Processed data directories
PROCESSED_DIR = BASE_DIR / "processed_data"
EXTRACTED_TEXT_DIR = PROCESSED_DIR / "extracted_text"
EXTRACTED_TABLES_DIR = PROCESSED_DIR / "extracted_tables"
STRUCTURED_RECORDS_DIR = PROCESSED_DIR / "structured_records"

# Database directories
# NOTE: the live application (upload, search, filter, build_index) has always
# used data/chroma_db. This is now the single definition everyone imports.
DB_DIR = DATA_DIR
VECTOR_DB_DIR = DATA_DIR / "chroma_db"
COLLECTION_NAME = "wastewater_documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Uploaded files and their JSON sidecar metadata
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_METADATA_DIR = UPLOAD_DIR / ".metadata"

# Logging directory
LOG_DIR = BASE_DIR / "logs"
