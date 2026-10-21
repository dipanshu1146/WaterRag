
from pathlib import Path

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.extraction.pdf_table_extractor import extract_pdf_tables
from src.extraction.table_to_text import table_rows_to_text

from openpyxl import load_workbook

from src.ingestion.pipeline import read_text_file


TXT_DIR = Path("data/txt")
PDF_DIR = Path("data/uploads")


def load_text_documents(data_path=TXT_DIR):
    """Load plain-text documents."""
    documents = []
    data_folder = Path(data_path)

    if not data_folder.exists():
        return documents

    if data_folder.is_file():
        if data_folder.suffix.lower() != ".txt":
            return documents
        text_files = [data_folder]
    else:
        text_files = sorted(data_folder.glob("*.txt"))

    for file_path in text_files:
        text = read_text_file(file_path)  # tolerant decoding

        if not text.strip():
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "file_type": "txt",
                    "content_type": "text",
                },
            )
        )

    return documents






def load_excel_documents(data_path=PDF_DIR):
    """Load Excel STP records with correct headers and plant identity."""
    import re

    documents = []
    data_folder = Path(data_path)

    if not data_folder.exists():
        return documents

    if data_folder.is_file():
        if data_folder.suffix.lower() != ".xlsx":
            return documents
        excel_files = [data_folder]
    else:
        excel_files = sorted(data_folder.glob("*.xlsx"))

    for file_path in excel_files:
        workbook = load_workbook(
            filename=file_path,
            read_only=True,
            data_only=True,
        )

        try:
            for sheet in workbook.worksheets:
                # Each section has its own headers in row 4.
                header_row = next(
                    sheet.iter_rows(
                        min_row=4, max_row=4, values_only=True
                    ),
                    None,
                )
                if not header_row:
                    continue

                headers = [
                    str(value).strip() if value is not None else f"Column_{i + 1}"
                    for i, value in enumerate(header_row)
                ]

                current_stp_id = None
                current_stp_name = None

                for row_number, row in enumerate(
                    sheet.iter_rows(min_row=6, values_only=True),
                    start=6,
                ):
                    heading = row[1] if len(row) > 1 else None
                    heading_text = (
                        str(heading).strip() if heading is not None else ""
                    )

                    match = re.search(
                        r"\b([A-Z]{2,}-STP-\s*\d+)\b",
                        heading_text,
                        flags=re.IGNORECASE,
                    )

                    if match:
                        current_stp_id = re.sub(r"\s+", "", match.group(1).upper())
                        current_stp_name = heading_text
                        continue

                    if not any(
                        value is not None and str(value).strip()
                        for value in row
                    ):
                        continue

                    identity = []
                    if current_stp_id:
                        identity.append(f"STP ID: {current_stp_id}")
                    if current_stp_name:
                        identity.append(f"STP name: {current_stp_name}")

                    # Excel column ranges are one-based and inclusive.
                    sections = [
                        ("outlet", 14, 74),
                        ("inlet", 75, 134),
                        ("sludge", 135, 154),
                    ]

                    for sample_type, first_col, last_col in sections:
                        fields = []

                        for col_number in range(first_col, last_col + 1):
                            index = col_number - 1

                            if index >= len(row):
                                continue

                            value = row[index]
                            if value is None or not str(value).strip():
                                continue

                            fields.append(f"{headers[index]}: {value}")

                        # Keep general identifiers, dates and report details
                        # from columns A:M with the outlet sample record.
                        if sample_type == "outlet":
                            for index in range(min(13, len(row))):
                                value = row[index]
                                if value is not None and str(value).strip():
                                    fields.insert(
                                        index,
                                        f"{headers[index]}: {value}",
                                    )

                        if not fields:
                            continue


                        # Format outlet records that explicitly report no flow.
                        if sample_type == "outlet":
                            report_value = row[12] if len(row) > 12 else None

                        if(
                            report_value is not None
                            and "no flow" in str(report_value).strip().lower()
                        ):
                            fields = [
                            field
                            for field in fields
                            if not field.startswith("STP's Name:")
                            and not field.startswith("Report No. & date:")
                            ]

                            fields.append(f"Flow status: {str(report_value).strip()}")

                        
                        # Skip outlet headings and section labels that contain no sample data.
                        if sample_type == "outlet":
                            sample_values = [
                                row[index]
                                for index in range(13, min(74, len(row)))
                                if row[index] is not None and str(row[index]).strip()
                            ]

                            if not sample_values:
                                # Preserve "No Flow" status records from the outlet report column.
                                report_value = row[12] if len(row) > 12 else None

                                if not (
                                                           report_value is not None
                                    and "no flow" in str(report_value).strip().lower()
                                ):
                                    continue                   


                        record_text = "\n".join(
                            identity
                            + [f"Sample type: {sample_type}"]
                            + fields
                        )

                        documents.append(
                            Document(
                                page_content=record_text,
                                metadata={
                                    "source": str(file_path),
                                    "file_type": "xlsx",
                                    "content_type": "table_row",
                                    "sheet": sheet.title,
                                    "row_number": row_number,
                                    "stp_id": current_stp_id or "unknown",
                                    "stp_name": current_stp_name or "unknown",
                                    "sample_type": sample_type,
                                },
                            )
                        )
        finally:
            workbook.close()

    return documents






def load_pdf_documents(data_path=PDF_DIR):
    """Load PDF narrative text and individual table-row records."""
    documents = []
    data_folder = Path(data_path)

    if not data_folder.exists():
        return documents

    if data_folder.is_file():
        if data_folder.suffix.lower() != ".pdf":
            return documents
        pdf_files = [data_folder]
    else:
        pdf_files = sorted(data_folder.glob("*.pdf"))

    for file_path in pdf_files:

        # Extract narrative text while excluding characters
        # located inside detected table boundaries.
        with pdfplumber.open(file_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                tables_on_page = page.find_tables()

                def outside_tables(obj):
                    if obj.get("object_type") != "char":
                        return True

                    for table in tables_on_page:
                        x0, top, x1, bottom = table.bbox

                        inside_table = (
                            obj["x0"] >= x0
                            and obj["x1"] <= x1
                            and obj["top"] >= top
                            and obj["bottom"] <= bottom
                        )

                        if inside_table:
                            return False

                    return True

                narrative_page = page.filter(outside_tables)
                text = narrative_page.extract_text() or ""

                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": str(file_path),
                                "file_type": "pdf",
                                "page": page_number,
                                "content_type": "text",
                            },
                        )
                    )

        # Extract tables separately and convert each row to labelled text.
        tables = extract_pdf_tables(file_path)

        for table in tables:
            for record in table_rows_to_text(table):
                documents.append(
                    Document(
                        page_content=record["content"],
                        metadata={
                            **record,
                            "source": str(file_path),
                        },
                    )
                )

    return documents


def load_documents():
    """Load all currently supported text, PDF, and Excel records."""
    return (
        load_text_documents()
        + load_pdf_documents()
        + load_excel_documents()
    )


if __name__ == "__main__":
    documents = load_documents()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Number of documents/records: {len(documents)}")
    print(f"Number of chunks: {len(chunks)}")

    print("\n--- Record types ---")
    record_types = {}

    for document in documents:
        record_type = document.metadata.get("content_type", "unknown")
        record_types[record_type] = record_types.get(record_type, 0) + 1

    for record_type, count in record_types.items():
        print(f"{record_type}: {count}")

    print("\n--- First 5 chunks ---")

    for index, chunk in enumerate(chunks[:5], start=1):
        print("\n==============================")
        print("Chunk:", index)
        print("Source:", chunk.metadata.get("source"))
        print("Page:", chunk.metadata.get("page", "N/A"))
        print("Content type:", chunk.metadata.get("content_type"))
        print("==============================")
        print(chunk.page_content)
