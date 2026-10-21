
def table_rows_to_text(table: dict) -> list[dict]:
    """Convert extracted table rows into labelled text records."""

    headers = table["headers"]
    records = []

    for row in table["rows"]:
        # Ignore rows with no meaningful content
        if not any(str(value).strip() for value in row):
            continue

        # Map each column header to its corresponding value
        fields = []
        for header, value in zip(headers, row):
            header = " ".join(str(header).split())
            value = " ".join(str(value).split())

            if header and value:
                fields.append(f"{header}: {value}")

        if not fields:
            continue

        content = "\n".join(fields)

        records.append({
            "content": content,
            "source": table["source"],
            "file_type": table["file_type"],
            "page": table["page"],
            "table_number": table["table_number"],
            "content_type": "table_row",
        })

    return records
