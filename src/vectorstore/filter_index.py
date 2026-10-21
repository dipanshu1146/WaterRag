
import re
from pathlib import Path
from collections import Counter

from langchain_chroma import Chroma


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
from src.config import VECTOR_DB_DIR as DB_DIR  # single shared path
COLLECTION_NAME = "wastewater_documents"


# --------------------------------------------------
# 2. GENERAL HELPERS
# --------------------------------------------------

def normalize_key(value):
    """Normalize field names for reliable matching."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def parse_record(content, metadata):
    """Convert a labelled table row into a structured record."""
    fields = {}

    for line in (content or "").splitlines():
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        fields[normalize_key(key)] = value.strip()

    return {
        "fields": fields,
        "content": content or "",
        "source": metadata.get("source", "Unknown"),
        "page": metadata.get("page", "N/A"),
        "table_number": metadata.get("table_number", "N/A"),
    }


def get_field(fields, *possible_names):
    """Return the first non-empty matching field."""
    for name in possible_names:
        value = fields.get(normalize_key(name))
        if value is not None and str(value).strip():
            return str(value).strip()

    return ""


def get_town(record):
    """Return the town or district associated with a record."""
    fields = record.get("fields", {})

    return get_field(
        fields,
        "Town",
        "Name of District",
        "District",
        "City",
        "Plant Name",
        "Name of STP",
        "STP Name",
        "Name of CETP",
    )


def get_plant_name(record):
    """Return a specific plant name when the source provides one."""
    return get_field(
        record.get("fields", {}),
        "Name of STP",
        "STP Name",
        "Name of CETP",
        "Plant Name",
        "Name of Plant",
    )


def get_compliance(record):
    """Return the original compliance-status text."""
    return get_field(
        record.get("fields", {}),
        "Compliance status of STPs",
        "Compliance status of STP",
        "Compliance Status",
        "Status of Compliance",
    )


def is_non_operational(value):
    """
    Recognize explicit non-operational wording.

    Examples:
    Non-operational
    Non operational
    Not operational
    """
    text = str(value or "").casefold()

    return bool(
        re.search(
            r"\b(?:non[\s-]*operational|not[\s-]+operational)\b",
            text,
        )
    )



def get_operational_status(record):
    """
    Determine operational status from the operational-status field.

    If the operational-status field conflicts with the compliance
    text, return 'Conflicting source information'.
    """
    fields = record.get("fields", {})

    status = get_field(
        fields,
        "Operational status of STP",
        "Operationa l status of STP",
        "Operational Status",
        "Operational status",
        "Status of STP",
    )

    compliance = get_compliance(record)

    status_says_non_operational = is_non_operational(status)
    compliance_says_non_operational = is_non_operational(compliance)

    status_says_operational = bool(
        re.search(r"\boperational\b", status, re.IGNORECASE)
    ) and not status_says_non_operational

    # The two source fields explicitly disagree.
    if (
        status_says_operational
        and compliance_says_non_operational
    ):
        return "Conflicting source information"

    # Explicit non-operational status in either field.
    if status_says_non_operational or compliance_says_non_operational:
        return "Non-operational"

    if status_says_operational:
        return "Operational"

    return "Unknown"





def is_non_operational_record(record):
    """
    Return True only when the interpreted status is explicitly
    non-operational, not when the source fields conflict.
    """
    return get_operational_status(record) == "Non-operational"


def get_capacity(record):
    """Extract capacity in MLD from supported source field names."""
    value = get_field(
        record.get("fields", {}),
        "Existing STP Capacity (MLD)",
        "Existing STP Capacity",
        "STP Capacity (MLD)",
        "Capacity of STP (In MLD)",
        "Capacity of STP",
        "Capacity (MLD)",
        "Capacity of the Plant",
        "Capacity",
    )

    match = re.search(
        r"-?\d+(?:\.\d+)?",
        value.replace(",", ""),
    )

    return float(match.group()) if match else None


# --------------------------------------------------
# 3. COMPLIANCE HELPERS
# --------------------------------------------------


def is_non_complying(value):
    text = str(value or "")

    return bool(
        re.search(
            r"\b(?:non[\s-]*(?:complying|compliant|compliance)"
            r"|not[\s-]+(?:complying|compliant)"
            r"|failing\s+to\s+comply)\b",
            text,
            re.IGNORECASE,
        )
    )



def is_complying(value):
    text = str(value or "")

    return (
        bool(re.search(r"\bcomplying\b", text, re.IGNORECASE))
        and not is_non_complying(text)
    )


# --------------------------------------------------
# 4. LOAD TABLE ROWS FROM CHROMA
# --------------------------------------------------

def load_all_table_rows():
    """Load all indexed table rows without generating embeddings."""
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_DIR),
    )

    collection = vectorstore._collection
    records = []
    offset = 0
    batch_size = 500

    while True:
        batch = collection.get(
            where={"content_type": "table_row"},
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )

        documents = batch.get("documents") or []
        metadatas = batch.get("metadatas") or []

        for content, metadata in zip(documents, metadatas):
            records.append(parse_record(content, metadata or {}))

        offset += len(documents)

        if len(documents) < batch_size:
            break

    return records


# --------------------------------------------------
# 5. FACILITY CLASSIFICATION
# --------------------------------------------------

def get_facility_type(record):
    """Classify records as STP, CETP, or Unknown."""
    content = str(record.get("content", "")).casefold()
    fields = record.get("fields", {})

    field_names = {
        normalize_key(key)
        for key in fields
    }

    field_values = " ".join(
        str(value).casefold()
        for value in fields.values()
    )

    text = f"{content} {field_values}"

    # Check CETP first to avoid misclassification.
    if (
        "nameofcetp" in field_names
        or "cetpcapacity" in field_names
        or "common effluent treatment plant" in text
    ):
        return "CETP"

    stp_fields = (
        "nameofstp",
        "stpname",
        "stpcapacity",
        "existingstpcapacity",
        "capacityofstpinmld",
    )

    if any(
        any(term in name for term in stp_fields)
        for name in field_names
    ):
        return "STP"

    if (
        "sewage treatment plant" in text
        or "name of stp" in text
    ):
        return "STP"

    return "Unknown"


# --------------------------------------------------
# 6. RECORD-TYPE CLASSIFICATION
# --------------------------------------------------

def is_under_construction_record(record):
    """Identify project records using project-specific fields."""
    fields = record.get("fields", {})

    project_fields = (
        "Status of Tender",
        "Actual/Likely date of start work",
        "Actual/Likely date of Completion",
        "Status Yet to Start of Work",
    )

    return any(
        get_field(fields, field_name)
        for field_name in project_fields
    )


def is_aggregate_record(record):
    """Detect summaries describing several plants."""
    compliance = get_compliance(record)
    content = str(record.get("content", ""))
    text = f"{compliance} {content}"

    return bool(
        re.search(
            r"\b\d+\s*STPs?\s*(?:are|is|were|was)\b",
            text,
            re.IGNORECASE,
        )
    )


def get_record_type(record):
    """Classify individual plants, summaries, and projects."""
    if is_under_construction_record(record):
        return "Under-construction project"

    if is_aggregate_record(record):
        return "Town-level summary"

    plant_name = get_plant_name(record)
    capacity = get_capacity(record)
    compliance = get_compliance(record)

    if plant_name:
        return "Named plant record"

    if (
        get_town(record)
        and capacity is not None
        and compliance
        and not is_summary_record(record)
    ):
        return "Town-level or other record"

    return "Unclassified record"


def is_summary_record(record):
    """Identify aggregate rows and table headers."""
    fields = record.get("fields", {})

    sr_no = str(fields.get("srno", "")).strip().casefold()
    town = get_town(record).strip().casefold()

    aggregate_labels = {
        "total",
        "sub-total",
        "subtotal",
        "grand total",
    }

    if sr_no in aggregate_labels or town in aggregate_labels:
        return True

    if town == "not identified":
        return True

    normalized_fields = {
        normalize_key(key)
        for key in fields
    }

    header_terms = (
        "capacityoftheplant",
        "physicalprogress",
        "statusofid",
        "completiontimeline",
    )

    header_matches = sum(
        1
        for key in normalized_fields
        if any(term in key for term in header_terms)
    )

    if header_matches >= 2:
        return True

    if not get_town(record) and get_capacity(record) is None:
        return True

    return False


def get_canonical_town(record):
    """
    Return the town name rather than a longer location description.

    Examples:
    Rewari -> Rewari
    Rewari Road, Narnaul -> Narnaul
    Near Chinkara Rest House, Rewari Road, Narnaul -> Narnaul
    """
    raw_town = get_town(record).strip()

    if not raw_town:
        return ""

    town = raw_town.split(",")[-1].strip()

    invalid_values = {
        "total",
        "total capacity",
        "capacity",
        "not identified",
        "unknown",
        "n/a",
        "na",
    }

    if town.casefold() in invalid_values:
        return ""

    if re.search(
        r"\b(total|capacity|subtotal|grand total)\b",
        town,
        re.IGNORECASE,
    ):
        return ""

    return town


# --------------------------------------------------
# 7. TOWN MATCHING
# --------------------------------------------------

def find_requested_town(query, records):
    """Find a canonical town name mentioned in the question."""
    towns = {
        get_canonical_town(record)
        for record in records
        if get_canonical_town(record)
    }

    # Match the longest town name first.
    for town in sorted(towns, key=len, reverse=True):
        if re.search(
            r"(?<!\w)" + re.escape(town) + r"(?!\w)",
            query,
            re.IGNORECASE,
        ):
            return town

    return None


# --------------------------------------------------
# 8. STRUCTURED FILTERING
# --------------------------------------------------

def filter_records(query, records):
    query_lower = query.casefold()



    asks_conflicting_status = any(
        phrase in query_lower
        for phrase in (
            "conflicting status",
            "conflicting operational status",
            "status conflict",
            "contradictory status",
            "uncertain operational status",
        )
    )


    asks_for_stp = bool(
        re.search(r"\bstps?\b", query_lower)
        or "sewage treatment" in query_lower
    )

    asks_under_construction = any(
        phrase in query_lower
        for phrase in (
            "under construction",
            "under-construction",
            "upcoming stp",
            "upcoming plant",
            "stps being constructed",
            "stps under construction",
        )
    )

    
    asks_non_compliant = is_non_complying(query_lower)


    asks_non_operational = bool(
        re.search(
            r"\bnon[\s-]*operational\b|\bnot[\s-]+operational\b",
            query_lower,
        )
    )

    asks_operational = (
        "operational" in query_lower
        and not asks_non_operational
    )

    asks_compliant = (
        "complying" in query_lower
        and not asks_non_compliant
    )

    capacity_match = re.search(
        r"(?:greater than|more than|above|over|less than|below|under)"
        r"\s*(\d+(?:\.\d+)?)\s*(?:mld)?",
        query_lower,
    )

    capacity_operator = None
    capacity_threshold = None

    if capacity_match:
        phrase = capacity_match.group(0)

        capacity_operator = (
            "greater"
            if any(
                term in phrase
                for term in (
                    "greater than",
                    "more than",
                    "above",
                    "over",
                )
            )
            else "less"
        )

        capacity_threshold = float(capacity_match.group(1))

    town = find_requested_town(query, records)
    results = []

    for record in records:
        # Remove totals and obvious headers.
        if is_summary_record(record):
            continue

        facility_type = get_facility_type(record)

        if asks_for_stp and facility_type != "STP":
            continue

        is_project = is_under_construction_record(record)

        # Project records appear only when the query asks for them.
        if is_project and not asks_under_construction:
            continue

        if asks_under_construction and not is_project:
            continue

        # Do not claim project records are operational.
        if asks_operational and is_project:
            continue

        town_name = get_canonical_town(record)
        compliance = get_compliance(record)
        operational = get_operational_status(record)
        capacity = get_capacity(record)

        if town and town_name.casefold() != town.casefold():
            continue

        if asks_non_compliant and not is_non_complying(compliance):
            continue

        if asks_compliant and not is_complying(compliance):
            continue

        # Use the dedicated helper to check both source status fields.
        
        # Return conflicting records only when explicitly requested.
        if asks_conflicting_status:
            if operational != "Conflicting source information":
                continue

        elif asks_non_operational:
            if not is_non_operational_record(record):
                continue

        elif asks_operational:
            if operational != "Operational":
                continue


        if capacity_operator:
            if capacity is None:
                continue

            if (
                capacity_operator == "greater"
                and capacity <= capacity_threshold
            ):
                continue

            if (
                capacity_operator == "less"
                and capacity >= capacity_threshold
            ):
                continue

        results.append(record)

    return results, town


# --------------------------------------------------
# 9. DATA QUALITY DIAGNOSTICS
# --------------------------------------------------

def inspect_operational_statuses(records):
    """Count status values and display missing-status examples."""
    statuses = Counter()

    for record in records:
        status = get_operational_status(record)
        label = status if status else "(blank or not recognized)"
        statuses[label] += 1

    print("\nOperational status values found:")

    for status, count in sorted(statuses.items()):
        print(f"{status}: {count}")

    
    missing = [
        record
        for record in records
        if (
            get_facility_type(record) == "STP"
            and not is_summary_record(record)
            and get_operational_status(record) == "Unknown"
            and not is_under_construction_record(record)
        )
    ]


    print("\nExamples of rows with missing operational status:")

    for index, record in enumerate(missing[:5], start=1):
        print(f"\n--- Example {index} ---")
        print(record.get("content", ""))
        print(f"Source: {record.get('source')}")
        print(f"Page: {record.get('page')}")
        print(f"Table: {record.get('table_number')}")


# --------------------------------------------------
# 10. MAIN PROGRAM
# --------------------------------------------------

def main():
    print("Loading all indexed table rows...")

    records = load_all_table_rows()

    table_counts = Counter(
        str(record.get("table_number", "Unknown"))
        for record in records
    )

    print("\nRecords by table:")
    for table, count in sorted(table_counts.items()):
        print(f"Table {table}: {count}")

    print(f"\nTable rows available for filtering: {len(records)}")

    facility_counts = Counter(
        get_facility_type(record)
        for record in records
    )

    print("\nFacility classification counts:")
    for facility_type, count in facility_counts.items():
        print(f"{facility_type}: {count}")

    unknown_records = [
        record
        for record in records
        if get_facility_type(record) == "Unknown"
    ]

    print("\nSample Unknown records:")

    for index, record in enumerate(unknown_records[:5], start=1):
        print(f"\n--- Unknown record {index} ---")
        print(record.get("content", ""))
        print(f"Page: {record.get('page')}")
        print(f"Table: {record.get('table_number')}")

    stp_records = [
        record
        for record in records
        if get_facility_type(record) == "STP"
    ]

    excluded_records = [
        record
        for record in stp_records
        if is_summary_record(record)
    ]

    valid_records = [
        record
        for record in stp_records
        if not is_summary_record(record)
    ]

    print("\nSTP classification validation:")
    print(f"STP-classified rows: {len(stp_records)}")
    print(f"Rows excluded as totals/headers: {len(excluded_records)}")
    print(f"Rows remaining after filtering: {len(valid_records)}")

    print("\nSTP record types among remaining rows:")
    type_counts = Counter(
        get_record_type(record)
        for record in valid_records
    )

    for record_type, count in type_counts.items():
        print(f"{record_type}: {count}")

    inspect_operational_statuses(records)

    query = input(
        "\nAsk a structured wastewater question: "
    ).strip()

    if not query:
        print("No query entered. Please try again.")
        return

    results, town = filter_records(query, records)

    print(f"\nMatching records: {len(results)}")

    if not results:
        print("No matching records found.")
        print(
            "Try changing the town, capacity, status, "
            "or project criteria."
        )
        return

    for index, record in enumerate(results, start=1):
        print("\n" + "=" * 55)
        print(f"Record {index}")
        print(f"Record type: {get_record_type(record)}")
        print(f"Facility type: {get_facility_type(record)}")
        print(f"Town/District: {get_town(record) or 'Not available'}")

        plant_name = get_plant_name(record)
        if plant_name:
            print(f"Plant name: {plant_name}")

        capacity = get_capacity(record)

        if capacity is not None:
            print(f"Capacity: {capacity:g} MLD")
        else:
            print("Capacity: Not available")

        operational = get_operational_status(record)
        compliance = get_compliance(record)

        print(f"Operational status: {operational}")
        print(f"Compliance status: {compliance or 'Not available'}")

        print(f"Source: {record.get('source', 'Not available')}")
        print(f"Page: {record.get('page', 'Not available')}")
        print(f"Table: {record.get('table_number', 'Not available')}")

        print("-" * 55)
        print(record.get("content", ""))


if __name__ == "__main__":
    main()
