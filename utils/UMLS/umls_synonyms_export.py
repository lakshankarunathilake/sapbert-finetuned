#!/usr/bin/env python3
"""
UMLS MeSH Category Synonym Exporter

Reads UMLS Metathesaurus MRCONSO.RRF and exports MeSH-only synonyms for:
- a given top-level MeSH category (Diseases, Chemicals and Drugs, etc.)
- or multiple categories at once

Category membership is determined via MeSH Tree Numbers stored in MRHIER.RRF:
- Diseases -> tree prefix 'C'
- Chemicals and Drugs -> tree prefix 'D'
etc.

Output CSV columns:
ID, Terms
Where Terms is a "||"-joined list of deduplicated synonyms.

Key features:
- Streaming write (no DataFrame)
- Uses MRHIER to whitelist CUIs by MeSH category
- Filter by language (default ENG)
- Deduplication (case & whitespace insensitive)
- Optional preference sorting (MH/PT/PN/ISPREF first)
"""

import argparse
import csv
import gzip
import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Generator
from tqdm import tqdm


# ----------------------------
# MeSH Category prefix mapping
# ----------------------------
MESH_CATEGORY_PREFIX = {
    "ANATOMY": "A",
    "ORGANISMS": "B",
    "DISEASES": "C",
    "CHEMICALS": "D",
    "CHEMICALS_AND_DRUGS": "D",
    "ANALYTICAL_DIAGNOSTIC_AND_THERAPEUTIC_TECHNIQUES_AND_EQUIPMENT": "E",
    "TECHNIQUES": "E",
    "PSYCHIATRY_AND_PSYCHOLOGY": "F",
    "PSYCHIATRY": "F",
    "PHENOMENA_AND_PROCESSES": "G",
    "PHENOMENA": "G",
    "DISCIPLINES_AND_OCCUPATIONS": "H",
    "DISCIPLINES": "H",
    "ANTHROPOLOGY_EDUCATION_SOCIOLOGY_AND_SOCIAL_PHENOMENA": "I",
    "SOCIAL": "I",
    "TECHNOLOGY_INDUSTRY_AGRICULTURE": "J",
    "TECHNOLOGY": "J",
    "HUMANITIES": "K",
    "INFORMATION_SCIENCE": "L",
    "NAMED_GROUPS": "M",
    "HEALTH_CARE": "N",
}


def normalize_category(cat: str) -> str:
    """Normalize user category input to uppercase underscore form."""
    return cat.strip().upper().replace("&", "AND").replace("-", "_").replace(" ", "_")


# ----------------------------
# File utilities
# ----------------------------
def get_file_lines(path: str) -> Optional[int]:
    """Approximate line count for progress bar."""
    try:
        if path.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
                return sum(1 for _ in f)
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return sum(1 for _ in f)
    except:
        return None


def open_file(path: str):
    """Open file (.gz or plain)."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore", newline="")
    return open(path, "r", encoding="utf-8", errors="ignore", newline="")


def normalize_term(term: str) -> str:
    """Fast normalization for deduplication."""
    return " ".join(term.split()).strip().lower()


# ----------------------------
# MRHIER processing (category filter)
# ----------------------------
def load_mesh_category_cuis(mrhier_path: str, categories: List[str]) -> Set[str]:
    """
    Load a set of CUIs belonging to one or more MeSH top-level categories.

    Uses MRHIER.RRF and checks HCD field (field 7) for tree numbers beginning with:
    - C (Diseases)
    - D (Chemicals and Drugs)
    etc.
    """
    prefixes: Set[str] = set()

    for cat in categories:
        key = normalize_category(cat)

        if key not in MESH_CATEGORY_PREFIX:
            valid = sorted(MESH_CATEGORY_PREFIX.keys())
            raise ValueError(
                f"Unknown category '{cat}'. Valid normalized options include:\n"
                + "\n".join(valid)
            )

        prefixes.add(MESH_CATEGORY_PREFIX[key])

    cuis: Set[str] = set()
    total_lines = get_file_lines(mrhier_path)

    with open_file(mrhier_path) as f:
        for line in tqdm(f, total=total_lines, desc="Reading MRHIER (MeSH category filter)", unit="lines"):
            parts = line.rstrip("\n").split("|")
            if len(parts) < 8:
                continue

            # MRHIER format:
            # CUI|AUI|CXN|PAUI|SAB|REL|PTR|HCD|...
            # Field 7 (HCD) contains the MeSH tree number (e.g., D10.570.755)
            cui = parts[0]
            sab = parts[4]
            hcd = parts[7]  # Changed from PTR (field 6) to HCD (field 7)

            if sab != "MSH":
                continue

            # Category match based on tree prefix (first letter of HCD)
            if hcd and hcd[0] in prefixes:
                cuis.add(cui)

    return cuis


# ----------------------------
# MRCONSO processing
# ----------------------------
def parse_mrconso_line(line: str) -> Optional[Dict[str, str]]:
    """Parse MRCONSO line into relevant fields."""
    parts = line.rstrip("\n").split("|")
    if len(parts) < 17:
        return None

    return {
        "CUI": parts[0],
        "LAT": parts[1],
        "ISPREF": parts[6],
        "SAB": parts[11],
        "TTY": parts[12],
        "CODE": parts[13],
        "STR": parts[14].strip(),
        "SUPPRESS": parts[16] if len(parts) > 16 else "",
    }


def should_include_term(
    record: Dict[str, str],
    lat_filter: Optional[str],
    allowed_sab: Optional[Set[str]],
    allowed_tty: Optional[Set[str]],
    include_suppressed: bool,
) -> bool:
    """Apply filters."""
    if lat_filter and record["LAT"] != lat_filter:
        return False
    if not include_suppressed and record["SUPPRESS"] in {"O", "Y"}:
        return False
    if allowed_sab and record["SAB"] not in allowed_sab:
        return False
    if allowed_tty and record["TTY"] not in allowed_tty:
        return False
    return True


def process_identifier_terms(
    terms: List[Tuple[str, bool]],
    prefer_preferred_first: bool,
    min_terms: int,
) -> Optional[List[str]]:
    """Deduplicate and sort terms."""
    if not terms:
        return None

    seen: Dict[str, Tuple[str, bool]] = {}

    for original_term, is_preferred in terms:
        norm = normalize_term(original_term)
        if norm and norm not in seen:
            seen[norm] = (original_term, is_preferred)

    if len(seen) < min_terms:
        return None

    unique_terms = list(seen.values())

    if prefer_preferred_first:
        unique_terms.sort(key=lambda x: (not x[1], x[0].lower()))
    else:
        unique_terms.sort(key=lambda x: x[0].lower())

    return [t[0] for t in unique_terms]


def process_mrconso_streaming(
    file_path: str,
    lat_filter: Optional[str],
    allowed_sab: Optional[Set[str]],
    allowed_tty: Optional[Set[str]],
    include_suppressed: bool,
    prefer_preferred_first: bool,
    min_terms: int,
    max_rows: int,
    use_source_id: bool,
    allowed_cuis: Optional[Set[str]] = None,
) -> Generator[List[str], None, None]:
    """
    Process MRCONSO and yield rows:
    [identifier, term1, term2, ...]
    where identifier is CUI (default) or CODE (--use-source-id)
    """

    by_identifier: Dict[str, List[Tuple[str, bool]]] = defaultdict(list)
    total_lines = get_file_lines(file_path)

    rows_written = 0

    with open_file(file_path) as f:
        for line in tqdm(f, total=total_lines, desc="Reading MRCONSO", unit="lines"):
            record = parse_mrconso_line(line)
            if not record:
                continue

            # Category whitelist filter (based on CUI, even if output is CODE)
            if allowed_cuis and record["CUI"] not in allowed_cuis:
                continue

            if not should_include_term(record, lat_filter, allowed_sab, allowed_tty, include_suppressed):
                continue

            identifier = record["CODE"] if use_source_id else record["CUI"]
            if not identifier:
                continue

            # Improved preferred term logic for MeSH:
            # ISPREF=Y or TTY in {MH,PT,PN}
            is_preferred = (
                record["ISPREF"] == "Y"
                or record["TTY"] in {"MH", "PT", "PN"}
            )

            by_identifier[identifier].append((record["STR"], is_preferred))

    for identifier in tqdm(by_identifier.keys(), desc="Processing identifiers", unit="id"):
        processed = process_identifier_terms(by_identifier[identifier], prefer_preferred_first, min_terms)
        if processed:
            yield [identifier] + processed
            rows_written += 1
            if max_rows and rows_written >= max_rows:
                break


# ----------------------------
# CSV output
# ----------------------------
def write_csv_streaming(output_path: str, rows_generator: Generator[List[str], None, None], use_source_id: bool):
    id_col = "SourceID" if use_source_id else "CUI"
    headers = [id_col, "Terms"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for row in tqdm(rows_generator, desc="Writing CSV", unit="rows"):
            if len(row) >= 2:
                identifier = row[0]
                # Convert all terms to lowercase before joining
                terms_lowercase = [term.lower() for term in row[1:]]
                terms = "||".join(terms_lowercase)
                writer.writerow([identifier, terms])


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Export MeSH-only synonyms filtered by MeSH tree category (Diseases, Chemicals, etc.)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("mrconso", help="Path to MRCONSO.RRF (optionally .gz)")
    parser.add_argument("-o", "--out", required=True, help="Output CSV file path")

    parser.add_argument("--mrhier", help="Path to MRHIER.RRF (optionally .gz) (required for --mesh-category)")
    parser.add_argument("--mesh-category",
                        help="MeSH category filter (comma-separated). Examples: Diseases, Chemicals, Anatomy")

    parser.add_argument("--lat", default="ENG", help="Language filter (default: ENG). Use 'ALL' for no filter.")
    parser.add_argument("--sab", default="MSH",
                        help="Comma-separated allowed SABs (default: MSH). Use 'ALL' for no filter.")
    parser.add_argument("--tty", help="Comma-separated allowed term types (e.g., 'MH,SY,PT')")
    parser.add_argument("--include-suppressed", action="store_true",
                        help="Include suppressed/obsolete entries (default: exclude)")
    parser.add_argument("--min-terms", type=int, default=1,
                        help="Only output IDs with at least this many terms (default: 1)")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="Stop after writing N IDs (0 = no limit)")
    parser.add_argument("--prefer-preferred-first", action="store_true",
                        help="Preferred terms first (ISPREF=Y or TTY in {MH,PT,PN})")
    parser.add_argument("--use-source-id", action="store_true",
                        help="Use CODE (MeSH Descriptor ID) instead of CUI as identifier")

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.mrconso):
        parser.error(f"MRCONSO file not found: {args.mrconso}")

    if args.mesh_category and not args.mrhier:
        parser.error("--mrhier is required when using --mesh-category")

    if args.mrhier and not os.path.exists(args.mrhier):
        parser.error(f"MRHIER file not found: {args.mrhier}")

    # Parse filters
    lat_filter = None if args.lat.upper() == "ALL" else args.lat.upper()

    allowed_sab = None
    if args.sab and args.sab.upper() != "ALL":
        allowed_sab = set(s.strip() for s in args.sab.split(","))

    allowed_tty = set(t.strip() for t in args.tty.split(",")) if args.tty else None

    # Category whitelist
    allowed_cuis = None
    if args.mesh_category:
        categories = [c.strip() for c in args.mesh_category.split(",") if c.strip()]
        print(f"Loading MeSH category CUIs for: {categories}")
        allowed_cuis = load_mesh_category_cuis(args.mrhier, categories)
        print(f"Loaded {len(allowed_cuis)} CUIs matching categories {categories}")

    print(f"\nProcessing MRCONSO: {args.mrconso}")
    print(f"Output: {args.out}")
    print(f"Language: {lat_filter or 'ALL'}")
    print(f"SAB filter: {allowed_sab or 'ALL'}")
    print(f"TTY filter: {allowed_tty or 'ALL'}")
    print(f"Use source ID: {args.use_source_id}")
    print(f"Preferred-first: {args.prefer_preferred_first}")
    print(f"Include suppressed: {args.include_suppressed}")
    print(f"Min terms: {args.min_terms}")
    print(f"Max rows: {args.max_rows or 'unlimited'}")
    if args.mesh_category:
        print(f"MeSH category filter: {args.mesh_category}")

    rows_gen = process_mrconso_streaming(
        file_path=args.mrconso,
        lat_filter=lat_filter,
        allowed_sab=allowed_sab,
        allowed_tty=allowed_tty,
        include_suppressed=args.include_suppressed,
        prefer_preferred_first=args.prefer_preferred_first,
        min_terms=args.min_terms,
        max_rows=args.max_rows,
        use_source_id=args.use_source_id,
        allowed_cuis=allowed_cuis,
    )

    write_csv_streaming(args.out, rows_gen, args.use_source_id)

    print(f"\n✅ Done. Output written to: {args.out}")


if __name__ == "__main__":
    main()
