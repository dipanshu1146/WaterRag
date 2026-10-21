

import re
import os
from datetime import datetime

from fastapi import APIRouter, HTTPException

from pydantic import BaseModel, Field

from langchain_groq import ChatGroq

from src.vectorstore.filter_index import (

    load_all_table_rows,

    filter_records,

    is_non_complying,

)

from src.vectorstore.search_index import create_vectorstore

router = APIRouter(prefix="/api", tags=["Question Answering"])

def invoke_llm(prompt):
    """Call the configured Groq chat model without loading a local LLM."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured on the backend.")
    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        temperature=0,
        api_key=api_key,
    )
    return llm.invoke(prompt)

class AskRequest(BaseModel):

    question: str = Field(min_length=3, max_length=1000)

def extract_value(content, labels):

    for label in labels:

        match = re.search(

            rf"(?im)^\s*{re.escape(label)}\s*:\s*([\d.]+)",

            content,

        )

        if match:

            try:

                return float(match.group(1))

            except ValueError:

                pass

    return None

def extract_text(content, labels):

    for label in labels:

        match = re.search(

            rf"(?im)^\s*{re.escape(label)}\s*:\s*(.*?)\s*$",

            content,

        )

        if match:

            return match.group(1).strip()

    return None

def normalize_compliance(content):

    status = extract_text(

        content,

        ["Compliance status of STPs", "Compliance Status"],

    )

    if not status:

        return "Unknown"

    normalized = re.sub(r"[-_]+", " ", status.casefold())

    normalized = re.sub(r"\s+", " ", normalized).strip()

    if "non complying" in normalized or "noncomplying" in normalized:

        return "Non-complying"

    if "complying" in normalized:

        return "Complying"

    return "Unknown"

def normalize_operational_status(content):

    status = extract_text(

        content,

        [

            "Operational status of STP",

            "Operationa l status of STP",

        ],

    )

    compliance_text = extract_text(

        content,

        [

            "Compliance status of STPs",

            "Compliance Status",

        ],

    )

    status_normalized = (status or "").casefold()

    compliance_normalized = (compliance_text or "").casefold()

    # Detect non-operational statements in either source field.

    status_says_non_operational = bool(

        re.search(

            r"\bnon[\s-]*operational\b",

            status_normalized,

        )

    )

    compliance_says_non_operational = bool(

        re.search(

            r"\bnon[\s-]*operational\b",

            compliance_normalized,

        )

    )

    status_says_operational = bool(

        re.search(r"\boperational\b", status_normalized)

    ) and not status_says_non_operational

    compliance_says_operational = bool(

        re.search(r"\boperational\b", compliance_normalized)

    ) and not compliance_says_non_operational

    # If the two fields disagree, do not silently choose one.

    if (

        status_says_operational

        and compliance_says_non_operational

    ) or (

        status_says_non_operational

        and compliance_says_operational

    ):

        return "Conflicting source information"

    if status_says_non_operational or compliance_says_non_operational:

        return "Non-operational"

    if status_says_operational or compliance_says_operational:

        return "Operational"

    return "Unknown"

def build_record(source):

    content = source["content"]

    return {

        "serial_number": extract_text(content, ["Sr. No."]),

        "town": extract_text(content, ["Town", "District", "City"]),

        "existing_capacity_mld": extract_value(

            content,

            [

                "Existing STP Capacity (MLD)",

                "Existing STP Capacity",

                "STP Capacity (MLD)",

                "Capacity of STP (In MLD)",

                "Capacity of STP",

            ],

        ),

        "utilized_capacity_mld": extract_value(

            content,

            ["Capacity Being Utilized"],

        ),

        "operational_status": normalize_operational_status(content),

        "compliance_status": normalize_compliance(content),

        "source": source["source"],

        "page": source["page"],

        "table_number": source["table_number"],

    }

def calculate_summary(records):

    capacity_records = [

        r for r in records if r["existing_capacity_mld"] is not None

    ]

    utilized_records = [

        r for r in records if r["utilized_capacity_mld"] is not None

    ]

    return {

        "records_count": len(records),

        "total_existing_capacity_mld": round(

            sum(r["existing_capacity_mld"] for r in capacity_records), 3

        ),

        "total_utilized_capacity_mld": round(

            sum(r["utilized_capacity_mld"] for r in utilized_records), 3

        ),

        "records_with_capacity": len(capacity_records),

        "records_with_utilized_capacity": len(utilized_records),

        "operational_records": sum(

            r["operational_status"] == "Operational" for r in records

        ),

        "complying_records": sum(

            r["compliance_status"] == "Complying" for r in records

        ),

        "non_complying_records": sum(

            r["compliance_status"] == "Non-complying" for r in records

        ),

        "unknown_compliance_records": sum(

            r["compliance_status"] == "Unknown" for r in records

        ),

    }

def generate_answer(question, records, summary, town):

    q = question.casefold()

    count_question = bool(

        re.search(r"\b(how many|number of|count of)\b", q)

    )

    capacity_question = any(

        phrase in q

        for phrase in [

            "total capacity",

            "capacity in total",

            "total existing capacity",

            "combined capacity",

            "total utilized capacity",

            "total utilisation",

            "total utilization",

        ]

    )

    utilization_question = any(

        phrase in q

        for phrase in [

            "total utilized capacity",

            "total utilisation",

            "total utilization",

        ]

    )

    compliance_question = any(

        phrase in q

        for phrase in [

            "complying",

            "non-complying",

            "non complying",

            "compliance",

        ]

    )

    operational_question = any(

        phrase in q

        for phrase in [

            "operational",

            "non-operational",

            "non operational",

            "conflicting status",

            "conflicting operational status",

            "status conflict",

            "contradictory status",

            "uncertain operational status",

        ]

    )

    conflicting_status_question = any(

        phrase in q

        for phrase in [

           "conflicting status",

            "conflicting operational status",

            "status conflict",

           "contradictory status",

           "uncertain operational status",

       ]

    )

    location = f" in {town}" if town else ""

    # Exact counts come from the retrieved, filtered records.

    if count_question:

        count = len(records)

        if conflicting_status_question:

            count = sum(

                r["operational_status"] == "Conflicting source information"

                for r in records

            )

            label = "STPs with conflicting operational status"

        elif "non-complying" in q or "non complying" in q:

            count = sum(

                r["compliance_status"] == "Non-complying"

                for r in records

            )

            label = "non-complying STPs"

        elif "complying" in q:

            count = sum(

                r["compliance_status"] == "Complying"

                for r in records

            )

            label = "complying STPs"

        elif "operational" in q and (

            "non-operational" in q or "non operational" in q

        ):

            count = sum(

                r["operational_status"] == "Non-operational"

                for r in records

            )

            label = "non-operational STPs"

        elif "operational" in q:

            count = sum(

                r["operational_status"] == "Operational"

                for r in records

            )

            label = "operational STPs"

        else:

            label = "STPs"

        if count == 0:

            return f"No {label} were found{location} matching your criteria."

        if conflicting_status_question:

            noun = (

                "STP with conflicting operational status"

                if count == 1

                else "STPs with conflicting operational status"

            )

            return f"{count} {noun}{location} found."

        return (

            f"There {'is' if count == 1 else 'are'} {count} "

            f"{label}{location} matching your criteria."

        )

    # Return capacity totals only when the question asks for them.

    if capacity_question:

        if utilization_question:

            total = summary["total_utilized_capacity_mld"]

            label = "total utilized capacity"

            available = summary["records_with_utilized_capacity"]

        else:

            total = summary["total_existing_capacity_mld"]

            label = "total existing capacity"

            available = summary["records_with_capacity"]

        if available == 0:

            return f"The {label} is not available in the retrieved records."

        return (

            f"The {label}{location} is {total:g} MLD, "

            f"based on {available} records with capacity data."

        )

    # Answer questions specifically about compliance.

    if compliance_question:

        if "non-complying" in q or "non complying" in q:

            count = summary["non_complying_records"]

            label = "non-complying"

        elif "complying" in q:

            count = summary["complying_records"]

            label = "complying"

        else:

            return (

                f"Among {len(records)} retrieved records{location}, "

                f"{summary['complying_records']} are complying, "

                f"{summary['non_complying_records']} are non-complying, "

                f"and {summary['unknown_compliance_records']} have "

                "unknown compliance status."

            )

        return (

            f"{count} retrieved STP records{location} are {label}."

        )

    # Answer questions specifically about operational status.

    if operational_question:

        if conflicting_status_question:

            count = sum(

                r["operational_status"] == "Conflicting source information"

                for r in records

            )

            label = (

                "record with conflicting operational status"

                if count == 1

                else "records with conflicting operational status"

            )

        elif "non-operational" in q or "non operational" in q:

            count = sum(

                r["operational_status"] == "Non-operational"

                for r in records

            )

            label = (

                "non-operational STP"

                if count == 1

                else "non-operational STPs"

            )

        else:

            count = sum(

                r["operational_status"] == "Operational"

                for r in records

            )

            label = (

                "operational STP"

                if count == 1

                else "operational STPs"

            )

        return f"{count} {label} found{location}."

    # Use the LLM for broader questions that need an explanation.

    context = "\n".join(

        f"Record {i}: town={r['town']}, "

        f"serial number={r['serial_number']}, "

        f"existing capacity={r['existing_capacity_mld']} MLD, "

        f"utilized capacity={r['utilized_capacity_mld']} MLD, "

        f"operational status={r['operational_status']}, "

        f"compliance={r['compliance_status']}"

        for i, r in enumerate(records, start=1)

    )

    prompt = f"""

You are a wastewater treatment information assistant.

Answer the user's question directly in 1-3 sentences.

Use only the verified summary and retrieved records below.

Do not invent facts or change numerical values.

Do not repeat every summary statistic unless relevant.

Mention that the answer is based on retrieved records when appropriate.

Question: {question}

Town filter: {town or "Not identified"}

Verified summary: {summary}

Retrieved records:

{context}

"""

    response = invoke_llm(prompt)

    return response.content.strip()

def should_use_semantic_fallback(question):

    """

    Use semantic search for broad explanatory questions.

    Keep exact structured queries restricted to their filters.

    """

    q = question.casefold()

    structured_terms = [

        "stp",

        "sewage treatment plant",

        "operational",

        "non-operational",

        "non operational",

        "conflicting status",

        "status conflict",

        "contradictory status",

        "complying",

        "non-complying",

        "non complying",

        "compliance",

        "capacity",

        "mld",

        "under construction",

        "how many",

        "number of",

        "count of",

        "total capacity",

        "total utilization",

        "total utilisation",

    ]

    return not any(term in q for term in structured_terms)

STP_ID_PATTERN = re.compile(r"\b[A-Z]{2,5}-STP-\d{3,}\b", re.IGNORECASE)

def _line_value(content, label_pattern):
    """Return a value from a complete key:value line matching label_pattern."""
    match = re.search(
        rf"(?im)^\s*(?:{label_pattern})\s*:\s*(.*?)\s*$",
        content or "",
    )
    return match.group(1).strip() if match else None

def _parse_sample_date(value):
    """Parse dates used by the source workbook, e.g. 03.07.2026."""
    if not value:
        return None

    text = str(value).strip()
    for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(text[:10], fmt).date()
        except (ValueError, TypeError):
            continue

    # Also accept ISO-like values that include a time component.
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except (ValueError, TypeError):
        return None

def _parse_outlet_sample(content, metadata):
    """
    Parse only the Outlet of STP result columns.

    The labels are intentionally strict: 'BOD mg/l (10)' is an outlet
    column, whereas 'BOD (mg/l)' belongs to the separate inlet section.
    The generic 'BOD: 10' line on the main STP row is not a sample result.
    """
    date_text = _line_value(content, r"Collection\s+Date")
    sample_date = _parse_sample_date(date_text)

    # The workbook's outlet columns embed their reference values in the
    # column names. Do not match the differently named inlet columns.
    outlet_bod = _line_value(
        content, r"BOD\s+mg\s*/\s*l(?:\s*\([^)]*\))?"
    )
    outlet_cod = _line_value(
        content, r"COD\s+mg\s*/\s*l(?:\s*\([^)]*\))?"
    )
    outlet_tss = _line_value(
        content, r"TSS\s+mg\s*/\s*l(?:\s*\([^)]*\))?"
    )
    outlet_ph = _line_value(content, r"P[Hh]\s*\([^)]*\)")

    values = {
        "bod": outlet_bod,
        "cod": outlet_cod,
        "tss": outlet_tss,
        "ph": outlet_ph,
    }

    if not sample_date and not any(value is not None for value in values.values()):
        return None

    row_number = metadata.get("row_number", metadata.get("table_number"))
    try:
        row_number = int(row_number)
    except (TypeError, ValueError):
        row_number = None

    # Preserve source report identifiers and sampling context when present.
    report_number = _line_value(
        content,
        r"(?:report\s*(?:no\.?|number)(?:\s*(?:&|and)\s*date)?|report\s*no\.?\s*&\s*date|sample\s*(?:id|no\.?|number)|letter\s*(?:no\.?|number))",
    )
    sample_context = "bypass" if re.search(r"\bbypass\b", content or "", re.I) else "outlet"

    return {
        "sample_date": sample_date,
        "report_number": report_number,
        "sample_context": sample_context,
        **values,
        "source": metadata.get("source", "Unknown"),
        "sheet": metadata.get("sheet", "N/A"),
        "page": metadata.get("page", "N/A"),
        "table_number": metadata.get(
            "table_number", metadata.get("row_number", "N/A")
        ),
        "row_number": row_number,
        "content": content,
    }

def _parse_inlet_values(content):
    """Read values specifically labelled as inlet measurements."""
    # The workbook has both "pH:" on the main STP row and "PH:" in its
    # inlet group. Preserve that case distinction for this one label.
    inlet_ph_match = re.search(
        r"(?m)^\s*PH\s*:\s*(.*?)\s*$",
        content or "",
    )
    return {
        "bod": _line_value(content, r"BOD\s*\(\s*mg\s*/\s*l\s*\)"),
        "cod": _line_value(content, r"COD\s*\(\s*mg\s*/\s*l\s*\)"),
        "tss": _line_value(content, r"TSS\s*\(\s*mg\s*/\s*l\s*\)"),
        "ph": inlet_ph_match.group(1).strip() if inlet_ph_match else None,
    }


def _extract_location_scope(question):
    """Extract a short place phrase without swallowing the rest of the question."""
    matches = list(re.finditer(r"\b(?:in|at|within)\s+([^,.?!]+)", question, re.IGNORECASE))
    if not matches:
        return None
    candidate = matches[-1].group(1).strip()
    # Stop before common question wording, e.g. "Yamuna Nagar are marked..."
    candidate = re.split(
        r"\s+\b(?:are|is|were|was|have|has|had|show|showing|marked|reported|listed|available|for|with|that|which|whose|using|having)\b\s+",
        candidate, maxsplit=1, flags=re.IGNORECASE
    )[0]
    candidate = re.sub(r"^(?:all\s+)?(?:the\s+)?(?:stps?|sewage\s+treatment\s+plants)\s+", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\b(?:stps?|sewage\s+treatment\s+plants)\s*$", "", candidate, flags=re.IGNORECASE)
    candidate = candidate.strip(" \t\r\n,.;:")
    return candidate or None

def answer_excel_water_quality(question):
    """
    Answer structured STP water-quality questions from indexed Excel rows.

    Each STP has a base row and multiple dated outlet-sampling rows.
    This function joins continuation rows to the most recent STP heading
    by source, sheet and Excel row_number, then returns the latest dated
    outlet sample for each matching plant.
    """
    q = question.casefold()

    parameter_patterns = [
        ("BOD", "bod", r"\bbod\b"),
        ("COD", "cod", r"\bcod\b"),
        ("TSS", "tss", r"\btss\b"),
        ("pH", "ph", r"\bph\b"),
    ]
    requested = [
        (label, field)
        for label, field, pattern in parameter_patterns
        if re.search(pattern, q, re.IGNORECASE)
    ]
    # Route general water-quality/sample-result questions here too.
    # When no parameter is named, report all supported parameters available
    # in the parsed Excel sample rows rather than falling through to PDF RAG.
    asks_general_samples = bool(re.search(
        r"water[ -]?quality|sample results?|sampling results?|sample data|"
        r"water quality parameters|parameters recorded",
        q,
        re.IGNORECASE,
    ))
    if not requested and not asks_general_samples:
        return None
    if not requested:
        requested = [
            ("BOD", "bod"),
            ("COD", "cod"),
            ("TSS", "tss"),
            ("pH", "ph"),
        ]

    # For now, a generic BOD/COD/TSS question means latest outlet sample.
    # An explicit "inlet" question reads only the separately labelled inlet
    # values on the STP's base row.
    asks_inlet = bool(re.search(r"\b(inlet|influent)\b", q))
    id_match = STP_ID_PATTERN.search(question.upper())
    requested_id = id_match.group(0).upper() if id_match else None

    location_match = _extract_location_scope(question)

    vectorstore = create_vectorstore()
    collection = vectorstore._collection
    excel_data = collection.get(
        where={"file_type": "xlsx"},
        include=["documents", "metadatas"],
    )

    entries = []
    documents = excel_data.get("documents") or []
    metadatas = excel_data.get("metadatas") or []

    for index, (content, metadata) in enumerate(zip(documents, metadatas)):
        metadata = metadata or {}
        row_number = metadata.get("row_number", metadata.get("table_number"))
        try:
            order_number = int(row_number)
        except (TypeError, ValueError):
            order_number = 10**12 + index

        entries.append({
            "content": content or "",
            "metadata": metadata,
            "index": index,
            "order_number": order_number,
            "source": str(metadata.get("source", "Unknown")),
            "sheet": str(metadata.get("sheet", "N/A")),
        })

    entries.sort(
        key=lambda item: (
            item["source"],
            item["sheet"],
            item["order_number"],
            item["index"],
        )
    )

    plants = []
    current_group = None
    active_plant = None
    active_town = None

    for item in entries:
        group = (item["source"], item["sheet"])
        if group != current_group:
            current_group = group
            active_plant = None
            active_town = None

        content = item["content"]
        metadata = item["metadata"]

        name_match = re.search(
            r"(?im)^\s*STP['’]?s\s+Name\s*:\s*(.*?)\s*$",
            content,
        )
        stp_name = name_match.group(1).strip() if name_match else None
        id_in_row = STP_ID_PATTERN.search(content.upper())

        if id_in_row:
            row_stp_id = id_in_row.group(0).upper()

            # Indexed Excel continuation rows also contain the STP ID.
            # Keep rows for the active plant in its sample history.
            if active_plant and active_plant["stp_id"] == row_stp_id:
                sample = _parse_outlet_sample(content, metadata)
                if sample is not None and sample["sample_date"] is not None:
                    if any(
                        sample.get(field) is not None
                        for _, field in requested
                    ):
                        active_plant["samples"].append(sample)

                inlet = _parse_inlet_values(content)
                if any(
                    inlet.get(field) is not None
                    for field in ("bod", "cod", "tss", "ph")
                ):
                    if sample is not None and sample.get("sample_date") is not None:
                        active_plant["inlet_samples"].append({**sample, **inlet})

                continue

            plant = {
                "stp_id": id_in_row.group(0).upper(),
                "stp_name": stp_name or id_in_row.group(0).upper(),
                "section_town": active_town,
                "source": metadata.get("source", "Unknown"),
                "sheet": metadata.get("sheet", "N/A"),
                "base_page": metadata.get("page", "N/A"),
                "base_row": metadata.get(
                    "row_number", metadata.get("table_number", "N/A")
                ),
                "base_content": content,
                "samples": [],
                "inlet_samples": [],
            }
            plants.append(plant)
            active_plant = plant

            if asks_inlet:
                plant["inlet_values"] = _parse_inlet_values(content)
            continue

        # The same Excel column is also used for section labels such as
        # "Yamuna Nagar" and month labels such as "May, 2026".
        # A month label is not a new town/plant and must not break the
        # preceding STP's series of sample rows.
        if stp_name:
            is_month_heading = bool(
                re.fullmatch(
                    r"(?i)(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
                    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
                    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s*,?\s*\d{4}",
                    stp_name.strip(),
                )
            )
            if is_month_heading:
                continue

            if re.search(r"\bSTP\b", stp_name, re.IGNORECASE):
                # A non-ID plant heading is not allowed to inherit the prior
                # plant's samples.
                active_plant = None
            else:
                active_town = stp_name
                active_plant = None
            continue

        if active_plant is not None:
            sample = _parse_outlet_sample(content, metadata)
            if sample is not None and sample["sample_date"] is not None:
                if any(sample.get(field) is not None for _, field in requested):
                    active_plant["samples"].append(sample)
                inlet = _parse_inlet_values(content)
                if any(inlet.get(field) is not None for field in ("bod", "cod", "tss", "ph")):
                    active_plant["inlet_samples"].append({**sample, **inlet})

    matched_plants = []
    for plant in plants:
        if requested_id and plant["stp_id"] != requested_id:
            continue

        location_evidence = " ".join([
            plant.get("stp_name", ""),
            plant.get("section_town") or "",
        ]).casefold()
        if location_match and location_match.casefold() not in location_evidence:
            continue

        if asks_inlet:
            dated_inlet = [s for s in plant.get("inlet_samples", []) if s.get("sample_date")]
            latest = max(dated_inlet, key=lambda s: (s["sample_date"], s.get("row_number") or -1), default=None)
            values = latest or plant.get("inlet_values") or _parse_inlet_values(plant.get("base_content", ""))
            sample_date = latest["sample_date"].isoformat() if latest else None
            content = latest["content"] if latest else plant["base_content"]
            sample_row = latest["table_number"] if latest else plant["base_row"]
            sample_source = latest.get("source", plant["source"]) if latest else plant["source"]
            sample_page = latest.get("page", plant["base_page"]) if latest else plant["base_page"]
        else:
            dated_samples = [
                sample for sample in plant["samples"]
                if sample.get("sample_date") is not None
            ]
            latest = max(
                dated_samples,
                key=lambda sample: (
                    sample["sample_date"],
                    sample.get("row_number")
                    if sample.get("row_number") is not None else -1,
                ),
                default=None,
            )
            values = latest or {}
            sample_date = (
                latest["sample_date"].isoformat() if latest else None
            )
            content = (
                f"{plant['base_content']}\n"
                f"Latest outlet sample row:\n{latest['content']}"
                if latest else plant["base_content"]
            )
            sample_row = (
                latest["table_number"] if latest else plant["base_row"]
            )
            sample_source = plant["source"]
            sample_page = (
                latest["page"] if latest else plant["base_page"]
            )

        capacity_match = re.search(
            r"\b(\d+(?:\.\d+)?)\s*MLD\b",
            plant.get("stp_name", ""),
            re.IGNORECASE,
        )

        record = {
            "stp_id": plant["stp_id"],
            "stp_name": plant["stp_name"],
            "capacity_mld": (
                float(capacity_match.group(1)) if capacity_match else None
            ),
            "sample_section": "Inlet of STP" if asks_inlet else "Outlet of STP",
            "sample_date": sample_date,
            "bod": values.get("bod"),
            "cod": values.get("cod"),
            "tss": values.get("tss"),
            "ph": values.get("ph"),
            "source": sample_source,
            "page": sample_page,
            "table_number": sample_row,
        }

        record["_evidence_content"] = content
        record["_base_row"] = plant["base_row"]
        matched_plants.append(record)

    if not matched_plants:
        # Let existing PDF retrieval continue if there are no Excel plants
        # for this question.
        return None

    # De-duplicate by plant ID and source, retaining the record with the
    # newest sample date if a workbook was indexed more than once.
    deduplicated = {}
    for record in matched_plants:
        key = (record["stp_id"], str(record["source"]))
        old = deduplicated.get(key)
        if old is None or (record.get("sample_date") or "") > (old.get("sample_date") or ""):
            deduplicated[key] = record
    records = sorted(
        deduplicated.values(),
        key=lambda record: (
            record["stp_id"],
            str(record["source"]),
        ),
    )

    def display_value(value):
        if value is None or str(value).strip() == "":
            return "not available in the selected row"
        return str(value)

    include_capacity = bool(re.search(r"\b(capacity|mld)\b", q))
    # Keep broad, unfiltered summaries readable while preserving every
    # matched record and source in the API response.
    limit_answer = (
        asks_general_samples
        and not requested_id
        and not location_match
    )
    display_records = records[:10] if limit_answer else records

    lines = []
    for record in display_records:
        requested_values = [
            f"{label}: {display_value(record.get(field))}"
            for label, field in requested
        ]
        if include_capacity:
            capacity_text = (
                f"{record['capacity_mld']:g} MLD"
                if record.get("capacity_mld") is not None
                else "not available"
            )
            requested_values.insert(0, f"Capacity: {capacity_text}")
        values_text = "; ".join(requested_values)
        if record.get("sample_date"):
            parsed_answer_date = _parse_sample_date(record["sample_date"])
            date_label = (
                parsed_answer_date.strftime("%d.%m.%Y")
                if parsed_answer_date
                else record["sample_date"]
            )
        else:
            date_label = "date not recorded"
        lines.append(
            f"{record['stp_id']} — {record['stp_name']} "
            f"(sample date: {date_label}): {values_text}"
        )

    location_label = f" in {location_match}" if location_match else ""
    if asks_inlet:
        intro = (
            f"Values extracted from the worksheet's inlet section{location_label}. "
            "These are inlet measurements, not treated outlet results."
        )
    else:
        intro = (
            f"Latest available outlet sample results{location_label}. "
            "BOD, COD and TSS are reported in mg/L where the source provides values; "
            "each STP uses its most recent dated outlet row in the indexed workbook."
        )

    if limit_answer and len(records) > len(display_records):
        lines.append(
            f"...and {len(records) - len(display_records)} more STPs. "
            "The complete results and source evidence are included in the "
            "structured records and sources."
        )

    public_records = [
        {key: value for key, value in record.items() if not key.startswith("_")}
        for record in records
    ]
    sources = []
    for index, record in enumerate(records, start=1):
        sources.append({
            "source_id": index,
            "document": record["source"],
            "page": record["page"],
            "table_number": record["table_number"],
            "content": record["_evidence_content"],
        })

    return {
        "question": question,
        "answer": intro + "\n" + "\n".join(lines),
        "summary": {
            "records_count": len(public_records),
            "records_with_sample_date": sum(
                bool(record.get("sample_date")) for record in public_records
            ),
            "sample_section": "Inlet of STP" if asks_inlet else "Outlet of STP",
        },
        "records": public_records,
        "sources": sources,
    }


def answer_excel_status_and_dates(question):
    """Handle sample-date and sample-level compliance questions from Excel."""
    q = question.casefold()
    asks_date = bool(re.search(r"\b(latest|most recent)\b.*\b(sample date|sampling date|collection date)\b|\b(sample date|sampling date|collection date)\b", q))
    asks_noncomplying = bool(re.search(r"non[ -]?complying|compliance status", q))
    if not (asks_date or asks_noncomplying):
        return None

    id_match = STP_ID_PATTERN.search(question.upper())
    requested_id = id_match.group(0).upper() if id_match else None
    location = _extract_location_scope(question)

    collection = create_vectorstore()._collection
    data = collection.get(where={"file_type": "xlsx"}, include=["documents", "metadatas"])
    entries=[]
    for i,(content,meta) in enumerate(zip(data.get("documents") or [], data.get("metadatas") or [])):
        meta=meta or {}
        try: row=int(meta.get("row_number", meta.get("table_number", 10**12+i)))
        except (TypeError,ValueError): row=10**12+i
        entries.append((str(meta.get("source","Unknown")),str(meta.get("sheet","N/A")),row,i,content or "",meta))
    entries.sort(key=lambda x:(x[0],x[1],x[2],x[3]))
    plants=[]; active=None; group=None; town=None
    for source,sheet,row,i,content,meta in entries:
        if (source,sheet)!=group: group=(source,sheet); active=None; town=None
        heading=re.search(r"(?im)^\s*STP['’]?s\s+Name\s*:\s*(.*?)\s*$",content)
        name=heading.group(1).strip() if heading else None
        idrow=STP_ID_PATTERN.search(content.upper())
        if idrow:
            active={"stp_id":idrow.group(0).upper(),"stp_name":name or idrow.group(0).upper(),"town":town,"source":source,"page":meta.get("page","N/A"),"base_row":row,"samples":[]}
            plants.append(active); continue
        if name:
            if re.fullmatch(r"(?i)(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s*,?\s*\d{4}",name): continue
            if re.search(r"\bSTP\b",name,re.I): active=None
            else: town=name; active=None
            continue
        if active:
            sample=_parse_outlet_sample(content,meta)
            if sample and sample.get('sample_date'):
                # Prefer the explicitly labelled legal/official remarks field.
                # Fall back to the plain Remarks field only if it is absent.
                legal_remarks = _line_value(
                    content,
                    r"Remarks\s*\(\s*actual\s*/\s*legally\s*\)",
                )
                plain_remarks = _line_value(content, r"Remarks")
                remarks = legal_remarks or plain_remarks or ""
                # Remove repeated identical status phrases without discarding
                # distinct context such as a bypass sample.
                remarks = re.sub(r"(?i)\b(non[ -]?complying)(?:\s+\1)+\b", r"\1", remarks).strip()
                sample['remarks'] = remarks
                active['samples'].append(sample)

    # Remove only exact duplicate source records. Keep samples with the same
    # STP/date but different report identifiers, remarks, or measurements.
    for plant in plants:
        unique_samples = []
        seen = set()
        for sample in plant["samples"]:
            identity = (
                plant["stp_id"],
                sample.get("sample_date"),
                (sample.get("report_number") or "").strip().casefold(),
                sample.get("sample_context"),
                sample.get("bod"), sample.get("cod"), sample.get("tss"), sample.get("ph"),
                (sample.get("remarks") or "").strip().casefold(),
            )
            if identity in seen:
                continue
            seen.add(identity)
            unique_samples.append(sample)
        plant["samples"] = unique_samples

    selected=[]
    for plant in plants:
        if requested_id and plant['stp_id']!=requested_id: continue
        evidence=(plant['stp_name']+' '+str(plant.get('town') or '')).casefold()
        if location and location.casefold() not in evidence: continue
        dated=plant['samples']
        latest=max(dated,key=lambda x:(x['sample_date'],x.get('row_number') or -1),default=None)
        if asks_noncomplying:
            # Status must be explicit in the source remarks, not inferred from numbers.
            marked=[x for x in dated if re.search(r"non[ -]?complying",x.get('remarks',''),re.I)]
            if marked:
                for sample in marked:
                    selected.append({"stp_id":plant['stp_id'],"stp_name":plant['stp_name'],"sample_date":sample['sample_date'].isoformat(),"compliance_status":"Non-complying","sample_section":"Outlet of STP","sample_context":sample.get("sample_context", "outlet"),"report_number":sample.get("report_number"),"bod":sample.get('bod'),"cod":sample.get('cod'),"tss":sample.get('tss'),"remarks":sample.get("remarks", ""),"source":plant['source'],"page":sample.get('page','N/A'),"table_number":sample.get('table_number','N/A'),"_content":sample.get('content','')})
        elif latest:
            selected.append({"stp_id":plant['stp_id'],"stp_name":plant['stp_name'],"sample_date":latest['sample_date'].isoformat(),"report_number":latest.get('report_number'),"sample_context":latest.get('sample_context', 'outlet'),"source":plant['source'],"page":latest.get('page','N/A'),"table_number":latest.get('table_number','N/A'),"_content":latest.get('content','')})
        else:
            selected.append({"stp_id":plant['stp_id'],"stp_name":plant['stp_name'],"sample_date":None,"source":plant['source'],"page":plant['page'],"table_number":plant['base_row'],"_content":plant['stp_name']})

    # For status questions, an empty result means no explicit Excel remark was found.
    if asks_noncomplying and not selected:
        return {"question":question,"answer":"No explicit non-complying remark was found in the indexed Excel outlet sample rows for the requested scope. This does not prove that all STPs comply.","summary":{"records_count":0,"sample_section":"Outlet of STP"},"records":[],"sources":[]}
    if not selected: return None
    selected.sort(key=lambda x:x['stp_id'])
    records=[]; sources=[]
    for idx,r in enumerate(selected,1):
        public={k:v for k,v in r.items() if not k.startswith('_')}; records.append(public)
        sources.append({"source_id":idx,"document":r['source'],"page":r['page'],"table_number":r['table_number'],"content":r['_content']})
    if asks_noncomplying:
        distinct_stps = len({r["stp_id"] for r in records})
        answer=(f"Found {len(records)} distinct outlet sample record(s) explicitly marked Non-complying across {distinct_stps} STP(s) in the indexed Excel workbook.\n"
                + "\n".join(
                    f"{r['stp_id']} — {r['stp_name']} | date: {r['sample_date']} | report: {r.get('report_number') or 'not stated'} | context: {r.get('sample_context', 'outlet')} | BOD: {r.get('bod') or 'not stated'} mg/L | COD: {r.get('cod') or 'not stated'} mg/L | TSS: {r.get('tss') or 'not stated'} mg/L | remark: {r.get('remarks') or 'Non-complying'} | source row/table: {r.get('table_number', 'N/A')}"
                    for r in records))
    else:
        answer="Latest available dated outlet sample date for each STP in the indexed Excel workbook. Missing dates are reported as not available.\n"+'\n'.join(f"{r['stp_id']} — {r['stp_name']}: {r['sample_date'] or 'No dated sample found'}" for r in records)
    return {"question":question,"answer":answer,"summary":{"records_count":len(records),"distinct_stps":len({r["stp_id"] for r in records}),"records_with_sample_date":sum(bool(r.get('sample_date')) for r in records),"sample_section":"Outlet of STP"},"records":records,"sources":sources}



def _excel_workbook_summary(question):
    """Answer explicit uploaded TXT/XLSX questions from their own indexed records."""
    q = question.casefold()

    asks_txt = bool(re.search(
        r"\b(?:uploaded\s+)?(?:text|txt)\s+document\b|\buploaded\s+txt\b",
        q,
    ))
    asks_excel = bool(re.search(
        r"\b(?:uploaded\s+)?(?:excel|xlsx|spreadsheet|workbook)\b",
        q,
    ))

    if not asks_txt and not asks_excel:
        return None

    collection = create_vectorstore()._collection
    file_type = "txt" if asks_txt else "xlsx"
    data = collection.get(
        where={"file_type": file_type},
        include=["documents", "metadatas"],
    )

    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    items = []

    for content, metadata in zip(documents, metadatas):
        metadata = metadata or {}
        source = str(metadata.get("source", "Unknown"))

        # Restrict this handler to files uploaded through the application.
        normalized_source = source.replace("\\", "/")
        if "data/uploads/" not in normalized_source:
            continue

        if asks_txt and metadata.get("content_type") != "text":
            continue

        items.append({
            "content": content or "",
            "source": source,
            "page": metadata.get("page", "N/A"),
            "table_number": metadata.get(
                "row_number", metadata.get("table_number", "N/A")
            ),
            "sheet": metadata.get("sheet", ""),
            "stp_id": metadata.get("stp_id", ""),
            "stp_name": metadata.get("stp_name", ""),
            "sample_type": metadata.get("sample_type", ""),
        })

    if not items:
        return {
            "question": question,
            "answer": (
                f"No indexed records were found for an uploaded "
                f"{'text' if asks_txt else 'Excel'} document."
            ),
            "summary": {"records_count": 0},
            "records": [],
            "sources": [],
        }

    if asks_txt:
        # A request about the text document should use its own text, not PDF rows.
        item = items[0]
        content = item["content"]

        def field(labels):
            for label in labels:
                match = re.search(
                    rf"(?im)^\s*{re.escape(label)}\s*:\s*(.*?)\s*$",
                    content,
                )
                if match:
                    return match.group(1).strip()
            return None

        fields = [
            ("Facility", field(["Test Facility", "Facility", "Company Name"])),
            ("Location", field(["Location", "District", "Address"])),
            ("Wastewater generation", field([
                "Wastewater Generation", "Wastewater Generation (KLD)",
                "Wastewater generated",
            ])),
            ("Treatment technology", field([
                "Treatment Technology", "Technology",
            ])),
            ("Treatment stage", field([
                "Treatment Stage", "Treatment Level",
            ])),
            ("Reuse", field([
                "Reuse", "Reuse Method", "Treated Wastewater Reuse",
            ])),
        ]

        answer = "Details extracted from the uploaded text document:\n" + "\n".join(
            f"{label}: {value or 'Not specified in this document'}"
            for label, value in fields
        )

        records = [{
            "content": content,
            "source": item["source"],
            "page": item["page"],
            "table_number": item["table_number"],
        }]
        sources = [{
            "source_id": 1,
            "document": item["source"],
            "page": item["page"],
            "table_number": item["table_number"],
            "content": content,
        }]

        return {
            "question": question,
            "answer": answer,
            "summary": {"records_count": 1, "file_type": "txt"},
            "records": records,
            "sources": sources,
        }

    # For broad workbook questions, report workbook coverage and sample rows.
    # Keep the returned evidence tied to Excel, not the general PDF table index.
    sheets = sorted({item["sheet"] for item in items if item["sheet"]})
    stp_ids = {item["stp_id"] for item in items if item["stp_id"]}
    samples = [
        item for item in items
        if re.search(
            r"(?im)^\s*(?:Collection Date|Sample Date)\s*:",
            item["content"],
        )
    ]

    preview = samples[:8] if samples else items[:8]
    answer_lines = [
        f"The uploaded Excel workbook has {len(items)} indexed records.",
        f"Distinct STP IDs identified in indexed metadata: {len(stp_ids)}.",
        "Sheets represented in the indexed records: "
        + (", ".join(sheets) if sheets else "not available"),
        f"Records containing a collection/sample date: {len(samples)}.",
        "Examples from the workbook:",
    ]

    for item in preview:
        content = item["content"]
        name = item["stp_name"] or "STP name not recorded"
        sample_type = item["sample_type"] or "type not specified"

        def value(pattern):
            match = re.search(pattern, content, re.I | re.M)
            return match.group(1).strip() if match else "not stated"

        date = value(r"^\s*(?:Collection Date|Sample Date)\s*:\s*(.+)$")
        ph = value(r"^\s*PH(?:\s*\([^)]*\))?\s*:\s*(.+)$")
        bod = value(r"^\s*BOD(?:\s+mg/l)?(?:\s*\([^)]*\))?\s*:\s*(.+)$")
        cod = value(r"^\s*COD(?:\s+mg/l)?(?:\s*\([^)]*\))?\s*:\s*(.+)$")
        tss = value(r"^\s*TSS(?:\s+mg/l)?(?:\s*\([^)]*\))?\s*:\s*(.+)$")
        remark = value(r"^\s*Remarks(?:\s*\([^)]*\))?\s*:\s*(.+)$")

        answer_lines.append(
            f"- {name} | {sample_type} | date: {date} | "
            f"pH: {ph} | BOD: {bod} | COD: {cod} | TSS: {tss} | "
            f"remark: {remark} | sheet: {item['sheet'] or 'not stated'} | "
            f"row: {item['table_number']}"
        )

    records = [
        {
            "content": item["content"],
            "source": item["source"],
            "page": item["page"],
            "table_number": item["table_number"],
            "sheet": item["sheet"],
            "stp_id": item["stp_id"],
            "stp_name": item["stp_name"],
        }
        for item in preview
    ]
    sources = [
        {
            "source_id": i,
            "document": item["source"],
            "page": item["page"],
            "table_number": item["table_number"],
            "content": item["content"],
            "sheet": item["sheet"],
        }
        for i, item in enumerate(preview, start=1)
    ]

    return {
        "question": question,
        "answer": "\n".join(answer_lines),
        "summary": {
            "records_count": len(items),
            "distinct_stps": len(stp_ids),
            "dated_sample_records": len(samples),
            "sheets": sheets,
            "file_type": "xlsx",
        },
        "records": records,
        "sources": sources,
    }



# ---------------------------------------------------------------------------
# General (all content types) retrieval-based answering
# ---------------------------------------------------------------------------

from src.vectorstore.general_search import (  # noqa: E402
    best_excerpt,
    display_name,
    general_search,
)

_EARLY_FILE_HINTS = [
    (r"\b(?:uploaded\s+)?(?:text|txt)\s+(?:document|file)\b|\buploaded\s+txt\b", ["txt"]),
    (r"\bcsv\b", ["csv"]),
    (r"\b(?:word\s+document|docx)\b", ["docx"]),
]
_ALL_FILE_HINTS = _EARLY_FILE_HINTS + [
    (r"\b(?:uploaded\s+)?(?:excel|xlsx|spreadsheet|workbook)\b", ["xlsx", "spreadsheet"]),
    (r"\bpdf\b", ["pdf"]),
]
_EXCEL_SUMMARY_INTENT = re.compile(
    r"\b(?:summar\w*|overview|coverage|how many (?:records|rows|sheets)|"
    r"what(?:'s| is) in)\b"
)
_FACILITY_PROFILE_TERMS = re.compile(
    r"wastewater\s+generat|treatment\s+(?:technolog|stage|level)|technolog|\breuse\b|re-use"
)


def _file_type_hint(question, hints=None):
    """File types the question explicitly names ("the CSV", "uploaded text file")."""
    q = question.casefold()
    for pattern, file_types in (hints or _ALL_FILE_HINTS):
        if re.search(pattern, q):
            return file_types
    return None


def _compose_answer(question, hits):
    """Answer from retrieved chunks; extractive fallback if the LLM is down."""
    context = "\n\n".join(
        f"[{i}] ({display_name(hit['source'])}"
        + (f", page {hit['page']}" if hit.get("page") else "")
        + f")\n{hit['content'][:1500]}"
        for i, hit in enumerate(hits[:4], start=1)
    )
    prompt = (
        "Answer the question using ONLY the context below. Quote exact values "
        "(names, numbers, dates) as written. If the context does not contain "
        "the answer, say that you cannot find it in the uploaded documents.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    try:
        reply = invoke_llm(prompt)
        text = (getattr(reply, "content", reply) or "").strip()
        if text:
            return text
    except Exception:
        pass

    top = hits[0]
    page = f", page {top['page']}" if top.get("page") else ""
    return (
        f"Most relevant passage from {display_name(top['source'])}{page}:\n"
        + best_excerpt(question, top["content"])
    )


def answer_with_general_retrieval(question, file_types=None, min_coverage=0.0):
    """Retrieve over ALL content types and answer. None when nothing relevant."""
    try:
        hits = general_search(question, k=5, file_types=file_types)
    except Exception:
        return None

    hits = [hit for hit in hits if hit.get("coverage", 0.0) >= min_coverage]
    if not hits:
        return None

    def location(hit):
        return hit.get("table_number") or hit.get("row_number") or "N/A"

    return {
        "question": question,
        "answer": _compose_answer(question, hits),
        "summary": {"records_count": len(hits), "retrieval": "general"},
        "records": [
            {
                "content": hit["content"],
                "source": hit["source"],
                "page": hit.get("page") or "N/A",
                "table_number": location(hit),
            }
            for hit in hits
        ],
        "sources": [
            {
                "source_id": i,
                "document": display_name(hit["source"]),
                "page": hit.get("page") or "N/A",
                "table_number": location(hit),
                "content": hit["content"],
            }
            for i, hit in enumerate(hits, start=1)
        ],
    }


def answer_uploaded_file_question(question):
    """Questions that explicitly name an uploaded TXT/CSV/DOCX file, or ask for
    a workbook summary, are answered from THAT file type by relevance, not by
    returning the first matching file."""
    q = question.casefold()

    early = _file_type_hint(question, _EARLY_FILE_HINTS)
    if early:
        result = answer_with_general_retrieval(question, file_types=early)
        if result is not None:
            return result
        return {
            "question": question,
            "answer": (
                "I could not find content relevant to your question in the "
                f"uploaded {'/'.join(early).upper()} document(s)."
            ),
            "summary": {"records_count": 0},
            "records": [],
            "sources": [],
        }

    # Whole-workbook overviews keep the dedicated summary view; any other
    # Excel question falls through to the structured/general handlers.
    if (
        re.search(r"\b(?:excel|xlsx|spreadsheet|workbook)\b", q)
        and _EXCEL_SUMMARY_INTENT.search(q)
    ):
        return _excel_workbook_summary(question)

    return None


@router.post("/ask")

def ask_question(request: AskRequest):

    question = request.question.strip()

    try:
        # Explicit uploaded-file questions must not fall through to PDF retrieval.
        uploaded_file_answer = answer_uploaded_file_question(question)
        if uploaded_file_answer is not None:
            return uploaded_file_answer

        # Route sample dates and sample-level compliance remarks to Excel first.
        excel_status_answer = answer_excel_status_and_dates(question)
        if excel_status_answer is not None:
            return excel_status_answer

        # Excel water-quality questions must use dated sample rows.
        # This branch runs before exact-ID lookup so an exact-ID parameter
        # question receives measured sample data rather than base-row limits.
        excel_water_quality_answer = answer_excel_water_quality(question)
        if excel_water_quality_answer is not None:
            return excel_water_quality_answer

        # Prevent unrelated RAG answers for STP-specific sample queries.
        requested_match = re.search(r"\b[A-Z]{2,5}-STP-\d{3,}\b", question.upper())
        sample_terms = ("BOD", "COD", "TSS", "PH", "WATER QUALITY",
                        "SAMPLE RESULT", "SAMPLE RESULTS", "OUTLET", "INLET")
        if requested_match and any(term in question.upper() for term in sample_terms):
            requested_id = requested_match.group(0)
            return {
                "question": question,
                "answer": (
                    f"No dated water-quality sample results were found in the indexed "
                    f"Excel data for {requested_id}. The available indexed data does "
                    f"not provide sample values for this query."
                ),
                "summary": {
                    "records_count": 0,
                    "stp_id": requested_id,
                    "sample_results_found": False,
                },
                "records": [],
                "sources": [],
            }

        # Exact lookup for STP IDs in indexed Excel records.

        stp_id_match = re.search(

            r"\b[A-Z]{2,5}-STP-\d{3,}\b",

            question.upper(),

        )

        if stp_id_match:

            requested_id = stp_id_match.group(0)

            vectorstore = create_vectorstore()

            collection = vectorstore._collection

            excel_data = collection.get(

                where={"file_type": "xlsx"},

                include=["documents", "metadatas"],

            )

            exact_matches = []

            for content, metadata in zip(

                excel_data.get("documents", []),

                excel_data.get("metadatas", []),

            ):

                if requested_id in content.upper():

                    exact_matches.append({

                        "content": content,

                        "source": metadata.get("source", "Unknown"),

                        "page": metadata.get("page", "N/A"),

                        "table_number": metadata.get(

                            "table_number",

                            metadata.get("row_number", "N/A"),

                        ),

                    })

            if exact_matches:

                records = []

                for item in exact_matches:

                    content = item["content"]

                    name_match = re.search(

                        r"STP['’]?s Name:\s*(.*)",

                        content,

                        re.IGNORECASE,

                    )

                    capacity_match = re.search(

                        r"(?:Treatment Capacity|STP Capacity)"

                        r"(?:\s*\(MLD\))?:\s*([0-9]+(?:\.[0-9]+)?)"

                        r"|STP['’]?s Name:\s*.*?"

                        r"\b([0-9]+(?:\.[0-9]+)?)\s*MLD\b",

                        content,

                        re.IGNORECASE,

                    )

                    records.append({

                        "stp_id": requested_id,

                        "stp_name": (

                            name_match.group(1).strip()

                            if name_match else requested_id

                        ),

                        "capacity_mld": (

                            float(next(

                                value

                                for value in capacity_match.groups()

                                if value is not None

                            ))

                            if capacity_match else None

                        ),

                        "bod": (

                            re.search(r"^\s*BOD:\s*(.+)$", content, re.I | re.M).group(1).strip()

                            if re.search(r"^\s*BOD:\s*(.+)$", content, re.I | re.M)

                            else None

                        ),

                        "ph": (

                            re.search(r"^\s*pH:\s*(.+)$", content, re.I | re.M).group(1).strip()

                            if re.search(r"^\s*pH:\s*(.+)$", content, re.I | re.M)

                            else None

                        ),

                        "cod": (

                            re.search(r"^\s*COD:\s*(.+)$", content, re.I | re.M).group(1).strip()

                            if re.search(r"^\s*COD:\s*(.+)$", content, re.I | re.M)

                            else None

                        ),

                        "tss": (

                            re.search(r"^\s*TSS:\s*(.+)$", content, re.I | re.M).group(1).strip()

                            if re.search(r"^\s*TSS:\s*(.+)$", content, re.I | re.M)

                            else None

                        ),

                        "treatment_technology": (

                            re.search(r"\(([^()]*)\)\s*$", name_match.group(1)).group(1).strip()

                            if name_match and re.search(r"\(([^()]*)\)\s*$", name_match.group(1))

                            else None

                        ),

                        "source": item["source"],

                                           "page": item["page"],

                        "table_number": item["table_number"],

                    })

                return {

                    "question": question,

                    "answer": "; ".join(

                        f"{record['stp_name']}: " + "; ".join(

                            f"{label}: {record[field]}"

                            for label, field, keywords in [

                                ("Capacity", "capacity_mld", ["capacity", "how many mld"]),

                                ("BOD", "bod", ["bod"]),

                                                   ("pH", "ph", ["ph"]),

                                ("COD", "cod", ["cod"]),

                                ("TSS", "tss", ["tss"]),

                                ("Treatment technology", "treatment_technology",

                                 ["technology", "treatment method", "process"]),

                            ]

                            if record.get(field) is not None

                            and any(word in question.casefold() for word in keywords)

                        )

                        for record in records

                    ),

                    "summary": {

                        "records_count": len(records),

                    },

                    "records": records,

                    "sources": [

                        {

                            "source_id": i,

                            "document": item["source"],

                            "page": item["page"],

                            "table_number": item["table_number"],

                            "content": item["content"],

                        }

                        for i, item in enumerate(exact_matches, start=1)

                    ],

                }

        # Search indexed text records for a facility named in the question.
        # Keep the original table-based retrieval as the fallback.
        facility_match = re.search(
            r"(?i)\b(?:of|for|at|about)\s+(.+?)(?:\s+in\s+[A-Za-z][A-Za-z .-]*[?.,]?$|[?.,]?$)",
            question.strip(),
        )

        if facility_match:
            requested_name = facility_match.group(1).strip(" \\t.,?!")
            generic_words = {
                "the facility", "this facility", "the plant",
                "this plant", "wastewater", "the wastewater",
                "the stp", "the etp",
            }

            if requested_name and requested_name.casefold() not in generic_words:
                collection = create_vectorstore()._collection
                indexed = collection.get(
                    include=["documents", "metadatas"],
                )

                exact_text_records = []
                requested_normalized = " ".join(requested_name.casefold().split())

                for content, metadata in zip(
                    indexed.get("documents") or [],
                    indexed.get("metadatas") or [],
                ):
                    metadata = metadata or {}
                    content = content or ""

                    if metadata.get("content_type") != "text":
                        continue

                    # Match the requested facility name against indexed text.
                    normalized_content = " ".join(content.casefold().split())
                    if requested_normalized not in normalized_content:
                        continue

                    exact_text_records.append({
                        "content": content,
                        "source": metadata.get("source", "Unknown"),
                        "page": metadata.get("page", "N/A"),
                        "table_number": metadata.get("table_number", "N/A"),
                    })

                if exact_text_records and _FACILITY_PROFILE_TERMS.search(question.casefold()):
                    evidence = exact_text_records[0]
                    content = evidence["content"]

                    fields = [
                        ("Wastewater generation",
                         extract_text(content, ["Wastewater Generation",
                                                "Wastewater Generation (KLD)",
                                                "Wastewater generated"])),
                        ("Treatment technology",
                         extract_text(content, ["Treatment Technology",
                                                "Treatment technology used",
                                                "Technology"])),
                        ("Treatment stage",
                         extract_text(content, ["Treatment Stage",
                                                "Treatment Level",
                                                "Stage of Treatment"])),
                        ("Reuse method",
                         extract_text(content, ["Reuse",
                                                "Reuse Method",
                                                "Treated Wastewater Reuse",
                                                "Utilization of Treated Wastewater"])),
                    ]

                    details = [
                        f"{label}: {value if value else 'Not specified in the matched text'}"
                        for label, value in fields
                    ]
                    answer_text = (
                        f"According to the matched source, facility "
                        f"'{requested_name}' has the following details:\\n"
                        + "\\n".join(details)
                    )

                    return {
                        "question": question,
                        "answer": answer_text,
                        "summary": {
                            "records_count": len(exact_text_records),
                        },
                        "records": exact_text_records,
                        "sources": [
                            {
                                "source_id": i,
                                "document": item["source"],
                                "page": item["page"],
                                "table_number": item["table_number"],
                                "content": item["content"],
                            }
                            for i, item in enumerate(exact_text_records, start=1)
                        ],
                    }

        # General retrieval over ALL content types (TXT, PDF narrative, CSV,
        # DOCX, generic spreadsheets). Questions that look like STP-database
        # queries keep using the structured table path below.
        if should_use_semantic_fallback(question):
            general_answer = answer_with_general_retrieval(
                question, file_types=_file_type_hint(question)
            )
            if general_answer is not None:
                return general_answer

        all_rows = load_all_table_rows()

        matches, town = filter_records(question, all_rows)

        sources = [

            {

                "content": item.get("content", ""),

                "source": item.get("source", "Unknown"),

                "page": item.get("page", "N/A"),

                "table_number": item.get("table_number", "N/A"),

            }

            for item in matches[:100]

        ]

        # Use semantic fallback only for broad questions.

        if not sources and should_use_semantic_fallback(question):

            vectorstore = create_vectorstore()

            documents = vectorstore.similarity_search(

                question,

                k=5,

                filter={"content_type": "table_row"},

            )

            sources = [

                {

                    "content": document.page_content,

                    "source": document.metadata.get("source", "Unknown"),

                    "page": document.metadata.get("page", "N/A"),

                    "table_number": document.metadata.get(

                        "table_number", "N/A"

                    ),

                }

                for document in documents

            ]

        # Deduplicate records.

        unique_sources = []

        seen = set()

        for source in sources:

            key = (

                source["source"],

                str(source["page"]),

                str(source["table_number"]),

                source["content"],

            )

            if key not in seen:

                seen.add(key)

                unique_sources.append(source)

        sources = unique_sources

        if not sources:

            # Structured STP filters found nothing: try the uploaded documents.
            general_answer = answer_with_general_retrieval(
                question, file_types=_file_type_hint(question)
            )
            if general_answer is not None:
                return general_answer

            q = question.casefold()

            location_text = f" in {town}" if town else ""

            asks_non_complying = bool(

                re.search(

                    r"\bnon[\s-]*(?:complying|compliant|compliance)\b",

                    q,

                )

            )

            asks_non_operational = bool(

                re.search(

                    r"\bnon[\s-]+operational\b|\bnot[\s-]+operational\b",

                    q,

                )

            )

            asks_conflicting_status = any(

                phrase in q

                for phrase in (

                    "conflicting status",

                    "conflicting operational status",

                    "status conflict",

                    "contradictory status",

                    "uncertain operational status",

                )

            )

            # If no confirmed non-operational records were found,

            # check for records with conflicting source information.

            if asks_non_operational and not asks_conflicting_status:

                conflict_query = (

                    f"Show STPs with conflicting status in {town}"

                    if town

                    else "Show STPs with conflicting status"

                )

                conflict_matches, _ = filter_records(

                    conflict_query,

                    all_rows,

                )

                conflict_sources = [

                    {

                        "content": item.get("content", ""),

                        "source": item.get("source", "Unknown"),

                        "page": item.get("page", "N/A"),

                        "table_number": item.get("table_number", "N/A"),

                    }

                    for item in conflict_matches[:100]

                ]

                # Remove duplicate source rows.

                unique_conflict_sources = []

                conflict_seen = set()

                for source in conflict_sources:

                    key = (

                        source["source"],

                        str(source["page"]),

                        str(source["table_number"]),

                        source["content"],

                    )

                    if key not in conflict_seen:

                        conflict_seen.add(key)

                        unique_conflict_sources.append(source)

                conflict_sources = unique_conflict_sources

                if conflict_sources:

                    conflict_records = [

                        build_record(source)

                        for source in conflict_sources

                    ]

                    # Count only records that are actually conflicting.

                    conflict_records = [

                        record

                        for record in conflict_records

                        if record["operational_status"]

                        == "Conflicting source information"

                    ]

                    if conflict_records:

                        location_text = (

                            f" in {town}" if town else ""

                        )

                        count = len(conflict_records)

                        answer = (

                            "No confirmed non-operational STPs were "

                            f"found{location_text}. However, {count} "

                            "record"

                            f"{' has' if count == 1 else 's have'} "

                            "conflicting operational information."

                        )

                        return {

                            "question": question,

                            "answer": answer,

                            "summary": calculate_summary(

                                conflict_records

                            ),

                            "records": conflict_records,

                            "sources": [

                                {

                                    "source_id": i,

                                    "document": source["source"],

                                    "page": source["page"],

                                    "table_number": source["table_number"],

                                    "content": source["content"],

                                }

                                for i, source in enumerate(

                                    conflict_sources,

                                    start=1,

                                )

                                if build_record(source)[

                                    "operational_status"

                                ] == "Conflicting source information"

                            ],

                        }

            if asks_non_complying:

                evidence_matches, _ = filter_records(

                    f"STPs in {town}" if town else "STPs",

                    all_rows,

                )

                evidence_sources = [

                    {

                        "content": item.get("content", ""),

                        "source": item.get("source", "Unknown"),

                        "page": item.get("page", "N/A"),

                        "table_number": item.get("table_number", "N/A"),

                    }

                    for item in evidence_matches[:100]

                ]

                evidence_records = [

                    build_record(source)

                    for source in evidence_sources

                ]

                confirmed_non_complying = [

                    record

                    for record in evidence_records

                    if is_non_complying(record["compliance_status"])

                ]

                if confirmed_non_complying:

                    answer = (

                        f"{len(confirmed_non_complying)} confirmed "

                        f"non-complying STP(s) found{location_text}."

                    )

                    output_records = confirmed_non_complying

                else:

                    answer = (

                        "No confirmed non-complying STPs were found"

                        f"{location_text} in the indexed wastewater records. "

                        "Unknown or conflicting statuses are not counted "

                        "as non-complying."

                    )

                    output_records = evidence_records

                return {

                    "question": question,

                    "answer": answer,

                    "summary": (

                        calculate_summary(output_records)

                        if output_records else None

                    ),

                    "records": output_records,

                    "sources": [

                        {

                            "source_id": i,

                            "document": source["source"],

                            "page": source["page"],

                            "table_number": source["table_number"],

                            "content": source["content"],

                        }

                        for i, source in enumerate(

                            evidence_sources, start=1

                        )

                    ],

                }

            return {

                "question": question,

                "answer": (

                    "No relevant records were found in the indexed "

                    "wastewater documents."

                ),

                "summary": None,

                "records": [],

                "sources": [],

            }

        records = [build_record(source) for source in sources]

        summary = calculate_summary(records)

        q = question.casefold()
        asks_capacity_and_status = (
            bool(re.search(r"\bcapacity\b", q))
            and bool(re.search(
                r"\b(?:operational|operation|compliance|status|utili[sz])\b",
                q,
            ))
        )

        if asks_capacity_and_status and records:
            answer_lines = []
            for record in records:
                name = record.get("town") or record.get("stp_name") or "STP"
                capacity = record.get("existing_capacity_mld")
                utilized = record.get("utilized_capacity_mld")
                operational = record.get("operational_status") or "Unknown"
                compliance = record.get("compliance_status") or "Unknown"

                details = []
                if capacity is not None:
                    details.append(f"existing capacity: {capacity} MLD")
                if utilized is not None:
                    details.append(f"capacity utilized: {utilized} MLD")
                details.append(f"operational status: {operational}")
                details.append(f"compliance status: {compliance}")

                answer_lines.append(f"{name}: " + "; ".join(details))

            answer = "\n".join(answer_lines)
        else:
            answer = generate_answer(question, records, summary, town)

        return {

            "question": question,

            "answer": answer,

            "summary": summary,

            "records": records,

            "sources": [

                {

                    "source_id": i,

                    "document": source["source"],

                    "page": source["page"],

                    "table_number": source["table_number"],

                    "content": source["content"],

                }

                for i, source in enumerate(sources, start=1)

            ],

        }

    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=f"Unable to answer the question: {str(exc)}",

        ) from exc
