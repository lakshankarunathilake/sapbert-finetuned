#!/usr/bin/env python3
"""
UMLS MeSH Category Synonym Exporter

Reads UMLS Metathesaurus MRCONSO.RRF and exports synonyms with optional
semantic type, semantic group, definition, and abbreviation enrichment.

Output CSV columns (configurable):
CUI, [SemanticGroups], [SemanticTypes], [Definition], Terms

Key features:
- Streaming write (no DataFrame)
- Uses MRHIER to whitelist CUIs by MeSH category
- Uses MRSTY for semantic group/type filtering and enrichment
- Uses MRDEF for definitions
- Uses LRABR (SPECIALIST Lexicon) for abbreviations/acronyms
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

# ----------------------------
# Semantic Group → TUI mapping
# ----------------------------
SEMANTIC_GROUP_TUIS = {
    "ANATOMY": {
        "T017", "T029", "T023", "T030", "T031", "T022",
        "T025", "T026", "T018", "T021", "T024"
    },
    "CHEM": {
        "T116", "T195", "T123", "T122", "T103", "T120", "T104", "T200",
        "T111", "T196", "T126", "T131", "T125", "T129", "T130", "T197",
        "T119", "T124", "T114", "T109", "T115", "T121", "T192", "T110",
        "T127", "T020"
    },
    "DEVICE": {"T074", "T075", "T073", "T072", "T167", "T168", "T071"},
    "DISO": {
        "T047", "T048", "T191", "T046", "T184", "T049",
        "T050", "T190", "T019"
    },
    "FINDING":          {"T033"},
    "INJURY_POISONING": {"T037"},
    "LABPROC":          {"T059"},
    "PHYS": {
        "T039", "T040", "T041", "T042", "T043", "T044",
        "T045", "T038", "T032", "T201"
    },
}

# TUI → human-readable group label (for output columns)
TUI_TO_GROUP: Dict[str, str] = {}
for _group, _tuis in SEMANTIC_GROUP_TUIS.items():
    for _tui in _tuis:
        TUI_TO_GROUP[_tui] = _group

# Additional TUIs not in our filter groups but useful for labelling
_EXTRA_TUI_GROUP = {
    "T007": "LIVING_BEINGS", "T204": "LIVING_BEINGS", "T194": "LIVING_BEINGS",
    "T008": "LIVING_BEINGS", "T010": "LIVING_BEINGS", "T011": "LIVING_BEINGS",
    "T012": "LIVING_BEINGS", "T013": "LIVING_BEINGS", "T014": "LIVING_BEINGS",
    "T015": "LIVING_BEINGS", "T016": "LIVING_BEINGS",
    "T060": "PROCEDURES", "T065": "PROCEDURES", "T058": "PROCEDURES",
    "T063": "PROCEDURES", "T062": "PROCEDURES", "T061": "PROCEDURES",
    "T093": "ORGANIZATIONS", "T092": "ORGANIZATIONS", "T094": "ORGANIZATIONS",
    "T095": "ORGANIZATIONS",
    "T078": "CONCEPTS", "T079": "CONCEPTS", "T080": "CONCEPTS",
    "T081": "CONCEPTS", "T082": "CONCEPTS", "T083": "CONCEPTS",
    "T170": "CONCEPTS", "T185": "CONCEPTS", "T169": "CONCEPTS",
    "T001": "LIVING_BEINGS", "T002": "LIVING_BEINGS", "T004": "LIVING_BEINGS",
    "T005": "LIVING_BEINGS", "T006": "LIVING_BEINGS",
}
TUI_TO_GROUP.update({k: v for k, v in _EXTRA_TUI_GROUP.items() if k not in TUI_TO_GROUP})


def normalize_category(cat: str) -> str:
    """Normalize user category input to uppercase underscore form."""
    return cat.strip().upper().replace("&", "AND").replace("-", "_").replace(" ", "_")


# ----------------------------
# File utilities
# ----------------------------
def get_file_lines(path: str) -> Optional[int]:
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
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore", newline="")
    return open(path, "r", encoding="utf-8", errors="ignore", newline="")


def normalize_term(term: str) -> str:
    return " ".join(term.split()).strip().lower()


# ----------------------------
# MRHIER processing (category filter)
# ----------------------------
def load_mesh_category_cuis(mrhier_path: str, categories: List[str]) -> Set[str]:
    """Load CUIs belonging to one or more MeSH top-level categories via MRHIER.RRF."""
    prefixes: Set[str] = set()
    for cat in categories:
        key = normalize_category(cat)
        if key not in MESH_CATEGORY_PREFIX:
            raise ValueError(
                f"Unknown category '{cat}'. Valid options:\n"
                + "\n".join(sorted(MESH_CATEGORY_PREFIX.keys()))
            )
        prefixes.add(MESH_CATEGORY_PREFIX[key])

    cuis: Set[str] = set()
    total_lines = get_file_lines(mrhier_path)

    with open_file(mrhier_path) as f:
        for line in tqdm(f, total=total_lines, desc="Reading MRHIER (MeSH category filter)", unit="lines"):
            parts = line.rstrip("\n").split("|")
            if len(parts) < 8:
                continue
            cui, sab, hcd = parts[0], parts[4], parts[7]
            if sab != "MSH":
                continue
            if hcd and hcd[0] in prefixes:
                cuis.add(cui)

    return cuis


# ----------------------------
# MRSTY processing (semantic types & group filter)
# ----------------------------
def load_cuis_by_semantic_group(mrsty_path: str, groups: List[str]) -> Set[str]:
    """
    Return the set of CUIs whose TUIs belong to any of the specified semantic groups.
    Valid groups: ANATOMY, CHEM, DEVICE, DISO, FINDING, INJURY_POISONING, LABPROC, PHYS
    """
    allowed_tuis: Set[str] = set()
    for group in groups:
        key = group.strip().upper()
        if key not in SEMANTIC_GROUP_TUIS:
            raise ValueError(
                f"Unknown semantic group '{group}'. Valid options:\n"
                + "\n".join(sorted(SEMANTIC_GROUP_TUIS.keys()))
            )
        allowed_tuis.update(SEMANTIC_GROUP_TUIS[key])

    cuis: Set[str] = set()
    total_lines = get_file_lines(mrsty_path)

    with open_file(mrsty_path) as f:
        for line in tqdm(f, total=total_lines, desc="Reading MRSTY (semantic group filter)", unit="lines"):
            parts = line.rstrip("\n").split("|")
            if len(parts) < 2:
                continue
            cui, tui = parts[0], parts[1]
            if tui in allowed_tuis:
                cuis.add(cui)

    return cuis


def load_mrsty(mrsty_path: str) -> Dict[str, Dict]:
    """
    Load MRSTY.RRF → CUI: { semantic_types: [...], semantic_groups: [...] }
    MRSTY format: CUI|TUI|STN|STY|ATUI|CVF|
    """
    result: Dict[str, Dict] = defaultdict(lambda: {"semantic_types": [], "semantic_groups": set()})
    total_lines = get_file_lines(mrsty_path)

    with open_file(mrsty_path) as f:
        for line in tqdm(f, total=total_lines, desc="Reading MRSTY (semantic types)", unit="lines"):
            parts = line.rstrip("\n").split("|")
            if len(parts) < 4:
                continue
            cui, tui, sty = parts[0], parts[1], parts[3]
            if not cui or not tui or not sty:
                continue

            entry = f"{tui}:{sty}"
            if entry not in result[cui]["semantic_types"]:
                result[cui]["semantic_types"].append(entry)
            result[cui]["semantic_groups"].add(TUI_TO_GROUP.get(tui, "UNKNOWN"))

    for cui in result:
        result[cui]["semantic_groups"] = sorted(result[cui]["semantic_groups"])

    return dict(result)


# ----------------------------
# MRDEF processing (definitions)
# ----------------------------
def load_mrdef(mrdef_path: str, allowed_sab: Optional[Set[str]] = None) -> Dict[str, str]:
    """
    Load MRDEF.RRF → CUI: definition string.
    Prefers MSH/NCI/SNOMEDCT_US/MEDLINEPLUS definitions; falls back to any.
    MRDEF format: CUI|AUI|ATUI|SATUI|SAB|DEF|SUPPRESS|CVF|
    """
    preferred_sabs = {"MSH", "NCI", "SNOMEDCT_US", "MEDLINEPLUS"}
    preferred: Dict[str, str] = {}
    fallback: Dict[str, str] = {}
    total_lines = get_file_lines(mrdef_path)

    with open_file(mrdef_path) as f:
        for line in tqdm(f, total=total_lines, desc="Reading MRDEF (definitions)", unit="lines"):
            parts = line.rstrip("\n").split("|")
            if len(parts) < 6:
                continue
            cui, sab, definition = parts[0], parts[4], parts[5].strip()
            suppress = parts[6] if len(parts) > 6 else ""

            if not definition or suppress in {"O", "Y"}:
                continue
            if allowed_sab and sab not in allowed_sab and sab not in preferred_sabs:
                continue

            if sab in preferred_sabs:
                if cui not in preferred:
                    preferred[cui] = definition
            else:
                if cui not in fallback:
                    fallback[cui] = definition

    return {**fallback, **preferred}


# ----------------------------
# LRABR processing (abbreviations)
# ----------------------------
def load_lrabr(lrabr_path: str) -> Dict[str, List[str]]:
    """
    Load SPECIALIST Lexicon LRABR file and return a mapping:
        normalized_expansion → [abbreviation, ...]

    LRABR format (pipe-separated):
        abbreviation | expansion | type | ...
    where type is typically 'a' (abbreviation) or 'A' (acronym).

    The expansion is matched case-insensitively against the terms collected
    from MRCONSO so that abbreviations can be appended as extra aliases.
    """
    # expansion (lowercased) → set of abbreviations
    expansion_to_abbrs: Dict[str, Set[str]] = defaultdict(set)
    total_lines = get_file_lines(lrabr_path)

    with open_file(lrabr_path) as f:
        for line in tqdm(f, total=total_lines, desc="Reading LRABR (abbreviations)", unit="lines"):
            parts = line.rstrip("\n").split("|")
            if len(parts) < 2:
                continue
            abbr      = parts[0].strip()
            expansion = parts[1].strip()
            if not abbr or not expansion:
                continue
            expansion_to_abbrs[normalize_term(expansion)].add(abbr)

    # Convert sets → sorted lists for deterministic output
    return {k: sorted(v) for k, v in expansion_to_abbrs.items()}


# ----------------------------
# MRCONSO processing
# ----------------------------
def parse_mrconso_line(line: str) -> Optional[Dict[str, str]]:
    parts = line.rstrip("\n").split("|")
    if len(parts) < 17:
        return None
    return {
        "CUI":      parts[0],
        "LAT":      parts[1],
        "ISPREF":   parts[6],
        "SAB":      parts[11],
        "TTY":      parts[12],
        "CODE":     parts[13],
        "STR":      parts[14].strip(),
        "SUPPRESS": parts[16] if len(parts) > 16 else "",
    }


def should_include_term(
    record: Dict[str, str],
    lat_filter: Optional[str],
    allowed_sab: Optional[Set[str]],
    allowed_tty: Optional[Set[str]],
    include_suppressed: bool,
) -> bool:
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
    sty_map: Optional[Dict[str, Dict]] = None,
    def_map: Optional[Dict[str, str]] = None,
    lrabr_map: Optional[Dict[str, List[str]]] = None,
) -> Generator[List[str], None, None]:
    """
    Process MRCONSO and yield rows:
    [identifier, semantic_groups, semantic_types, definition, term1, term2, ...]

    If lrabr_map is provided, abbreviations whose expansions match any collected
    term for a given entity are appended as additional aliases before deduplication.
    """
    by_identifier: Dict[str, List[Tuple[str, bool]]] = defaultdict(list)
    cui_for_identifier: Dict[str, str] = {}
    total_lines = get_file_lines(file_path)
    rows_written = 0

    with open_file(file_path) as f:
        for line in tqdm(f, total=total_lines, desc="Reading MRCONSO", unit="lines"):
            record = parse_mrconso_line(line)
            if not record:
                continue
            if allowed_cuis and record["CUI"] not in allowed_cuis:
                continue
            if not should_include_term(record, lat_filter, allowed_sab, allowed_tty, include_suppressed):
                continue

            identifier = record["CODE"] if use_source_id else record["CUI"]
            if not identifier:
                continue

            if use_source_id and identifier not in cui_for_identifier:
                cui_for_identifier[identifier] = record["CUI"]

            is_preferred = record["ISPREF"] == "Y" or record["TTY"] in {"MH", "PT", "PN"}
            by_identifier[identifier].append((record["STR"], is_preferred))

    for identifier in tqdm(by_identifier.keys(), desc="Processing identifiers", unit="id"):
        terms = list(by_identifier[identifier])

        # --- Append abbreviations from LRABR ---
        if lrabr_map:
            # Collect normalised forms of all terms already found for this entity
            existing_norms: Set[str] = {normalize_term(t) for t, _ in terms}
            abbrs_to_add: Set[str] = set()
            for norm in existing_norms:
                for abbr in lrabr_map.get(norm, []):
                    abbr_norm = normalize_term(abbr)
                    if abbr_norm and abbr_norm not in existing_norms:
                        abbrs_to_add.add(abbr)
            # Abbreviations are added as non-preferred (is_preferred=False)
            for abbr in sorted(abbrs_to_add):
                terms.append((abbr, False))

        processed = process_identifier_terms(terms, prefer_preferred_first, min_terms)
        if not processed:
            continue

        cui = cui_for_identifier.get(identifier, identifier) if use_source_id else identifier

        semantic_groups = ""
        semantic_types = ""
        if sty_map and cui in sty_map:
            sty_info = sty_map[cui]
            semantic_groups = "||".join(sty_info.get("semantic_groups", []))
            semantic_types  = "||".join(sty_info.get("semantic_types", []))

        definition = def_map.get(cui, "") if def_map else ""

        yield [identifier, semantic_groups, semantic_types, definition] + processed

        rows_written += 1
        if max_rows and rows_written >= max_rows:
            break


# ----------------------------
# CSV output
# ----------------------------
def write_csv_streaming(
    output_path: str,
    rows_generator: Generator[List[str], None, None],
    use_source_id: bool,
    include_sty: bool,
    include_def: bool,
):
    id_col = "SourceID" if use_source_id else "CUI"
    headers = [id_col]
    if include_sty:
        headers += ["SemanticGroups", "SemanticTypes"]
    if include_def:
        headers += ["Definition"]
    headers += ["Terms"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for row in tqdm(rows_generator, desc="Writing CSV", unit="rows"):
            identifier      = row[0]
            semantic_groups = row[1]
            semantic_types  = row[2]
            definition      = row[3]
            terms           = "||".join(term.lower() for term in row[4:])

            out_row = [identifier]
            if include_sty:
                out_row += [semantic_groups, semantic_types]
            if include_def:
                out_row += [definition]
            out_row += [terms]
            writer.writerow(out_row)


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Export UMLS synonyms with optional semantic type, group, definition, and abbreviation enrichment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export Diseases CUIs with semantic types and definitions
  python umls_synonyms_export.py MRCONSO.RRF --out output.csv \\
    --mrhier MRHIER.RRF --mrsty MRSTY.RRF --mrdef MRDEF.RRF \\
    --mesh-category Diseases --include-sty --include-def

  # Filter by semantic groups (DISO + CHEM) with enrichment
  python umls_synonyms_export.py MRCONSO.RRF --out output.csv \\
    --mrsty MRSTY.RRF --mrdef MRDEF.RRF \\
    --semantic-group DISO,CHEM --include-sty --include-def

  # All requested groups across all sources
  python umls_synonyms_export.py MRCONSO.RRF --out output.csv \\
    --mrsty MRSTY.RRF --mrdef MRDEF.RRF --sab ALL \\
    --semantic-group ANATOMY,CHEM,DEVICE,DISO,FINDING,INJURY_POISONING,LABPROC,PHYS \\
    --include-sty --include-def --prefer-preferred-first
        """,
    )

    parser.add_argument("mrconso", help="Path to MRCONSO.RRF (optionally .gz)")
    parser.add_argument("-o", "--out", required=True, help="Output CSV file path")

    # Source files
    parser.add_argument("--mrhier", help="Path to MRHIER.RRF (required for --mesh-category)")
    parser.add_argument("--mrsty",  help="Path to MRSTY.RRF (required for --include-sty or --semantic-group)")
    parser.add_argument("--mrdef",  help="Path to MRDEF.RRF (required for --include-def)")
    parser.add_argument("--lrabr",  help="Path to SPECIALIST Lexicon LRABR file for abbreviation/acronym enrichment")

    # Filters
    parser.add_argument("--mesh-category",
                        help="MeSH category filter (comma-separated). E.g.: Diseases,Chemicals")
    parser.add_argument("--semantic-group",
                        help=(
                            "Comma-separated semantic groups to include. "
                            "Options: ANATOMY, CHEM, DEVICE, DISO, FINDING, "
                            "INJURY_POISONING, LABPROC, PHYS"
                        ))

    # Enrichment flags
    parser.add_argument("--include-sty", action="store_true",
                        help="Add SemanticGroups and SemanticTypes columns (requires --mrsty)")
    parser.add_argument("--include-def", action="store_true",
                        help="Add Definition column (requires --mrdef)")

    # Term filters
    parser.add_argument("--lat", default="ENG",
                        help="Language filter (default: ENG). Use 'ALL' for no filter.")
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

    # Validate
    if not os.path.exists(args.mrconso):
        parser.error(f"MRCONSO file not found: {args.mrconso}")
    if args.mesh_category and not args.mrhier:
        parser.error("--mrhier is required when using --mesh-category")
    if args.include_sty and not args.mrsty:
        parser.error("--mrsty is required when using --include-sty")
    if args.include_def and not args.mrdef:
        parser.error("--mrdef is required when using --include-def")
    if args.semantic_group and not args.mrsty:
        parser.error("--mrsty is required when using --semantic-group")
    for label, path in [("MRHIER", args.mrhier), ("MRSTY", args.mrsty),
                        ("MRDEF", args.mrdef), ("LRABR", args.lrabr)]:
        if path and not os.path.exists(path):
            parser.error(f"{label} file not found: {path}")

    # Parse term filters
    lat_filter = None if args.lat.upper() == "ALL" else args.lat.upper()
    allowed_sab = None
    if args.sab and args.sab.upper() != "ALL":
        allowed_sab = set(s.strip() for s in args.sab.split(","))
    allowed_tty = set(t.strip() for t in args.tty.split(",")) if args.tty else None

    # Build CUI whitelist from MeSH category
    allowed_cuis: Optional[Set[str]] = None
    if args.mesh_category:
        categories = [c.strip() for c in args.mesh_category.split(",") if c.strip()]
        print(f"Loading MeSH category CUIs for: {categories}")
        allowed_cuis = load_mesh_category_cuis(args.mrhier, categories)
        print(f"  → {len(allowed_cuis):,} CUIs matched")

    # Intersect with semantic group whitelist
    if args.semantic_group:
        groups = [g.strip() for g in args.semantic_group.split(",") if g.strip()]
        print(f"Loading CUIs for semantic groups: {groups}")
        sty_cuis = load_cuis_by_semantic_group(args.mrsty, groups)
        print(f"  → {len(sty_cuis):,} CUIs matched")
        allowed_cuis = allowed_cuis & sty_cuis if allowed_cuis is not None else sty_cuis
        print(f"  → {len(allowed_cuis):,} CUIs after intersection")

    # Load enrichment maps
    sty_map = load_mrsty(args.mrsty) if args.include_sty else None
    if sty_map:
        print(f"Loaded semantic types for {len(sty_map):,} CUIs")

    def_map = load_mrdef(args.mrdef, allowed_sab) if args.include_def else None
    if def_map:
        print(f"Loaded definitions for {len(def_map):,} CUIs")

    lrabr_map = None
    if args.lrabr:
        lrabr_map = load_lrabr(args.lrabr)
        print(f"Loaded LRABR abbreviations for {len(lrabr_map):,} unique expansions")

    print(f"\nProcessing MRCONSO : {args.mrconso}")
    print(f"Output             : {args.out}")
    print(f"Language           : {lat_filter or 'ALL'}")
    print(f"SAB filter         : {allowed_sab or 'ALL'}")
    print(f"TTY filter         : {allowed_tty or 'ALL'}")
    print(f"Include STY        : {args.include_sty}")
    print(f"Include definition : {args.include_def}")
    print(f"Include LRABR abbr : {args.lrabr is not None}")
    print(f"Use source ID      : {args.use_source_id}")
    print(f"Preferred-first    : {args.prefer_preferred_first}")
    print(f"Min terms          : {args.min_terms}")
    print(f"Max rows           : {args.max_rows or 'unlimited'}")

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
        sty_map=sty_map,
        def_map=def_map,
        lrabr_map=lrabr_map,
    )

    write_csv_streaming(args.out, rows_gen, args.use_source_id, args.include_sty, args.include_def)

    print(f"\n✅ Done. Output written to: {args.out}")


if __name__ == "__main__":
    main()
