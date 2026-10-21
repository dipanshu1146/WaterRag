
from pathlib import Path
import re

import pdfplumber


def clean_cell(value) -> str:
    """Normalize whitespace in a cell."""
    if value is None:
        return ""
    return " ".join(str(value).split())


def is_ghaggar_under_construction_table(rows: list[list]) -> bool:
    """Identify the specific malformed Ghaggar Catchment table."""
    if not rows:
        return False

    header = " ".join(clean_cell(cell).lower() for cell in rows[0])
    return (
        "location" in header
        and "capacity o" in header
        and "deptt" in header
        and "physical" in header
        and "status of i&d" in header
        and "completio" in header
    )


def reconstruct_ghaggar_table(rows: list[list]) -> tuple[list[str], list[list[str]]]:
    """Reconstruct the seven Ghaggar Catchment project records."""
    headers = [
        "Sr. No.",
        "Location",
        "Capacity of plant (MLD)",
        "Department",
        "Physical progress (%)",
        "Shared I&D status (Ghaggar Catchment)",
        "Completion timeline",
    ]

    projects = []
    status_fragments = []

    # Skip the three header rows and the section-heading row.
    for raw_row in rows[3:]:
        row = [clean_cell(cell) for cell in raw_row]
        row += [""] * max(0, 21 - len(row))

        serial = row[1].rstrip(".").strip()

        # Collect the shared, vertically wrapped I&D note separately.
        if row[16]:
            status_fragments.append(row[16])

        if re.fullmatch(r"\d+", serial):
            projects.append({
                "serial": serial,
                "location": row[4],
                "capacity": row[7],
                "department": row[10],
                "progress": row[13],
                "timeline": row[19],
            })
        elif projects and row[4]:
            # A wrapped continuation of the preceding project's location.
            projects[-1]["location"] = (
                projects[-1]["location"] + " " + row[4]
            ).strip()

    shared_status = clean_cell(" ".join(status_fragments))

    # Remove accidental duplicate spaces and punctuation spacing.
    shared_status = re.sub(r"\s+([.,])", r"\1", shared_status)

    data = []
    for project in projects:
        data.append([
            project["serial"],
            project["location"],
            project["capacity"],
            project["department"],
            project["progress"],
            shared_status,
            project["timeline"],
        ])

    return headers, data


def extract_pdf_tables(file_path: str | Path) -> list[dict]:
    """Extract PDF tables while retaining page and table metadata."""
    path = Path(file_path)
    extracted_tables = []

    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            for table_number, table in enumerate(
                page.extract_tables(), start=1
            ):
                if not table:
                    continue

                raw_rows = [
                    [
                        str(cell).strip() if cell is not None else ""
                        for cell in row
                    ]
                    for row in table
                ]
                raw_rows = [
                    row for row in raw_rows
                    if any(clean_cell(cell) for cell in row)
                ]

                if not raw_rows:
                    continue

                # Special handling only for the malformed Ghaggar table.
                if is_ghaggar_under_construction_table(raw_rows):
                    headers, data_rows = reconstruct_ghaggar_table(raw_rows)
                else:
                    headers = [
                        clean_cell(cell.replace("\n", " "))
                        for cell in raw_rows[0]
                    ]
                    data_rows = []

                    for row in raw_rows[1:]:
                        cleaned = [clean_cell(cell) for cell in row]

                        if cleaned == headers:
                            continue

                        # Skip standalone section headings.
                        if (
                            cleaned[0]
                            and not cleaned[0].rstrip(".").isdigit()
                            and not any(cleaned[1:])
                        ):
                            continue

                        data_rows.append(cleaned)

                extracted_tables.append({
                    "source": path.name,
                    "file_type": "pdf",
                    "page": page_number,
                    "table_number": table_number,
                    "content_type": "table",
                    "headers": headers,
                    "rows": data_rows,
                })

    return extracted_tables
