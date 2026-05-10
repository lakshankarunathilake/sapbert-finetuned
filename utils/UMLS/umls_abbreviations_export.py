#!/usr/bin/env python3
"""
UMLS Entity Label Exporter

Exports a focused CSV with four columns per CUI:
    CUI | PreferredLabel | Abbreviations | Definition

- PreferredLabel : single best label (MH > PT > PN > ISPREF=Y > first ENG term)
- Abbreviations  : pipe-joined abbreviations/acronyms from SPECIALIST Lexicon LRABR
                   matched against the preferred label and all synonyms
- Definition     : best available definition (MSH > NCI > SNOMEDCT_US > MEDLINEPLUS > any)

Semantic group / MeSH category filtering is supported via --semantic-group and --mesh-category,
identical to umls_synonyms_export.py.

Usage examples
--------------
# Minimal (no filters, no abbreviations)
python umls_entity_labels_export.py MRCONSO.RRF --out output/labels.csv

# With semantic-group filter + LRABR abbreviations + definitions
python umls_entity_labels_export.py MRCONSO.RRF \\
  --out output/labels.csv \\
  --mrsty MRSTY.RRF \\
  --mrdef MRDEF.RRF \\
  --lrabr LRABR \\
  --semantic-group ANATOMY,CHEM,DEVICE,DISO,FINDING,INJURY_POISONING,LABPROC,PHYS \\
  --lat ENG --sab ALL
"""

import argparse
import csv
import gzip
import os
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Preferred TTY priority order (lower index = higher priority)
# ---------------------------------------------------------------------------
TTY_PRIORITY = ["MH", "PT", "PN", "ET", "SY"]


# ---------------------------------------------------------------------------
# Semantic Group → TUI sets  (same as umls_synonyms_export.py)
# ---------------------------------------------------------------------------
SEMANTIC_GROUP_TUIS: Dict[str, Set[str]] = {
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

MESH_CATEGORY_PREFIX: Dict[str, str] = {
    "ANATOMY": "A", "ORGANISMS": "B", "DISEASES": "C",
    "CHEMICALS": "D", "CHEMICALS_AND_DRUGS": "D",
    "TECHNIQUES": "E", "PSYCHIATRY": "F", "PHENOMENA": "G",
    "DISCIPLINES": "H", "SOCIAL": "I", "TECHNOLOGY": "J",
    "HUMANITIES": "K", "INFORMATION_SCIENCE": "L",
    "NAMED_GROUPS": "M", "HEALTH_CARE": "N",
}


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------
def open_file(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore", newline="")
    return open(path, "r", encoding="utf-8", errors="ignore", newline="")


def count_lines(path: str) -> Optional[int]:
    try:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception:
        return None


def normalize(text: str) -> str:
    return " ".join(text.split()).strip().lower()


# ---------------------------------------------------------------------------
# MRHIER — MeSH category whitelist
# ---------------------------------------------------------------------------
def load_mesh_cuis(mrhier_path: str, categories: List[str]) -> Set[str]:
    prefixes: Set[str] = set()
    for cat in categories:
        key = cat.strip().upper().replace("&", "AND").replace("-", "_").replace(" ", "_")
        if key not in MESH_CATEGORY_PREFIX:
            raise ValueError(f"Unknown MeSH category '{cat}'.")
        prefixes.add(MESH_CATEGORY_PREFIX[key])

    cuis: Set[str] = set()
    with open_file(mrhier_path) as f:
        for line in tqdm(f, total=count_lines(mrhier_path),
                         desc="MRHIER → MeSH category CUIs", unit="lines"):
            p = line.rstrip("\n").split("|")
            if len(p) < 8:
                continue
            if p[4] == "MSH" and p[7] and p[7][0] in prefixes:
                cuis.add(p[0])
    return cuis


# ---------------------------------------------------------------------------
# MRSTY — semantic group whitelist
# ---------------------------------------------------------------------------
def load_semantic_group_cuis(mrsty_path: str, groups: List[str]) -> Set[str]:
    allowed_tuis: Set[str] = set()
    for g in groups:
        key = g.strip().upper()
        if key not in SEMANTIC_GROUP_TUIS:
            raise ValueError(
                f"Unknown semantic group '{g}'. "
                f"Valid: {sorted(SEMANTIC_GROUP_TUIS.keys())}"
            )
        allowed_tuis.update(SEMANTIC_GROUP_TUIS[key])

    cuis: Set[str] = set()
    with open_file(mrsty_path) as f:
        for line in tqdm(f, total=count_lines(mrsty_path),
                         desc="MRSTY → semantic group CUIs", unit="lines"):
            p = line.rstrip("\n").split("|")
            if len(p) >= 2 and p[1] in allowed_tuis:
                cuis.add(p[0])
    return cuis


# ---------------------------------------------------------------------------
# MRDEF — definitions
# ---------------------------------------------------------------------------
PREFERRED_DEF_SABS = {"MSH", "NCI", "SNOMEDCT_US", "MEDLINEPLUS"}


def load_definitions(mrdef_path: str) -> Dict[str, str]:
    """CUI → best definition string."""
    preferred: Dict[str, str] = {}
    fallback:  Dict[str, str] = {}

    with open_file(mrdef_path) as f:
        for line in tqdm(f, total=count_lines(mrdef_path),
                         desc="MRDEF → definitions", unit="lines"):
            p = line.rstrip("\n").split("|")
            if len(p) < 6:
                continue
            cui, sab, defn = p[0], p[4], p[5].strip()
            suppress = p[6] if len(p) > 6 else ""
            if not defn or suppress in {"O", "Y"}:
                continue
            if sab in PREFERRED_DEF_SABS:
                preferred.setdefault(cui, defn)
            else:
                fallback.setdefault(cui, defn)

    return {**fallback, **preferred}   # preferred wins on key collision


# ---------------------------------------------------------------------------
# LRABR — abbreviations
# ---------------------------------------------------------------------------
def load_lrabr(lrabr_path: str) -> Dict[str, List[str]]:
    """normalized_expansion → sorted list of abbreviations."""
    result: Dict[str, Set[str]] = defaultdict(set)

    with open_file(lrabr_path) as f:
        for line in tqdm(f, total=count_lines(lrabr_path),
                         desc="LRABR → abbreviations", unit="lines"):
            p = line.rstrip("\n").split("|")
            if len(p) < 2:
                continue
            abbr, expansion = p[0].strip(), p[1].strip()
            if abbr and expansion:
                result[normalize(expansion)].add(abbr)

    return {k: sorted(v) for k, v in result.items()}


# ---------------------------------------------------------------------------
# MRCONSO — preferred label + all synonyms (for abbreviation matching)
# ---------------------------------------------------------------------------
def build_entity_table(
    mrconso_path: str,
    lat_filter: Optional[str],
    allowed_sab: Optional[Set[str]],
    allowed_cuis: Optional[Set[str]],
    include_suppressed: bool,
) -> Dict[str, Dict]:
    """
    Returns:
        {
          CUI: {
            "preferred_label": str,       # single best label
            "all_norms": set[str],        # normalized forms of all collected terms
          }
        }

    Preferred label selection priority:
        1. TTY in TTY_PRIORITY (ordered)
        2. ISPREF == "Y"
        3. First ENG term seen
    """

    # CUI → list of (str, tty, ispref)
    by_cui: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)

    with open_file(mrconso_path) as f:
        for line in tqdm(f, total=count_lines(mrconso_path),
                         desc="MRCONSO → collecting terms", unit="lines"):
            p = line.rstrip("\n").split("|")
            if len(p) < 17:
                continue

            cui      = p[0]
            lat      = p[1]
            ispref   = p[6]
            sab      = p[11]
            tty      = p[12]
            term     = p[14].strip()
            suppress = p[16]

            if allowed_cuis and cui not in allowed_cuis:
                continue
            if lat_filter and lat != lat_filter:
                continue
            if not include_suppressed and suppress in {"O", "Y"}:
                continue
            if allowed_sab and sab not in allowed_sab:
                continue
            if not term:
                continue

            by_cui[cui].append((term, tty, ispref))

    # Resolve preferred label per CUI
    result: Dict[str, Dict] = {}

    for cui, entries in tqdm(by_cui.items(), desc="Selecting preferred labels", unit="CUI"):
        # Build priority score: (tty_rank, ispref_rank)
        def score(entry: Tuple[str, str, str]) -> Tuple[int, int]:
            term, tty, ispref = entry
            tty_rank   = TTY_PRIORITY.index(tty) if tty in TTY_PRIORITY else len(TTY_PRIORITY)
            ispref_rank = 0 if ispref == "Y" else 1
            return (tty_rank, ispref_rank)

        best = min(entries, key=score)
        preferred_label = best[0]
        all_norms = {normalize(t) for t, _, _ in entries}

        result[cui] = {
            "preferred_label": preferred_label,
            "all_norms":       all_norms,
        }

    return result


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------
def export(
    entity_table: Dict[str, Dict],
    def_map: Optional[Dict[str, str]],
    lrabr_map: Optional[Dict[str, List[str]]],
    output_path: str,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["CUI", "PreferredLabel", "Abbreviations", "Definition"])

        for cui, info in tqdm(entity_table.items(), desc="Writing CSV", unit="rows"):
            preferred_label = info["preferred_label"]
            all_norms       = info["all_norms"]

            # Abbreviations: match any of the entity's terms against LRABR expansions
            abbreviations = ""
            if lrabr_map:
                abbr_set: Set[str] = set()
                for norm in all_norms:
                    for abbr in lrabr_map.get(norm, []):
                        if normalize(abbr) not in all_norms:   # skip if already a term
                            abbr_set.add(abbr)
                abbreviations = "||".join(sorted(abbr_set))

            definition = def_map.get(cui, "") if def_map else ""

            writer.writerow([cui, preferred_label, abbreviations, definition])

    print(f"\n✅ Done. {len(entity_table):,} entities written to: {output_path}")


# ---------------------------------------------------------------------------
# Abbreviation export (separate file)
# ---------------------------------------------------------------------------
def export_abbreviations(
    entity_table: Dict[str, Dict],
    lrabr_map: Dict[str, List[str]],
    output_path: str,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None

    rows_written = 0
    rows_skipped = 0

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["CUI", "PreferredLabel", "Abbreviations", "MatchedExpansions"])

        for cui, info in tqdm(entity_table.items(), desc="Matching abbreviations", unit="CUI"):
            preferred_label = info["preferred_label"]
            all_norms       = info["all_norms"]

            abbr_set: Set[str]           = set()
            matched_expansions: Set[str] = set()

            for norm in all_norms:
                for abbr in lrabr_map.get(norm, []):
                    if normalize(abbr) not in all_norms:
                        abbr_set.add(abbr)
                        matched_expansions.add(norm)

            # Always skip rows that have no matched abbreviations
            if not abbr_set:
                rows_skipped += 1
                continue

            writer.writerow([
                cui,
                preferred_label,
                "||".join(sorted(abbr_set)),
                "||".join(sorted(matched_expansions)),
            ])
            rows_written += 1

    print(f"\n✅ Done.")
    print(f"   Rows written (with abbreviations) : {rows_written:,}")
    print(f"   Rows skipped (no abbreviation)    : {rows_skipped:,}")
    print(f"   Output                            : {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Export CUI | PreferredLabel | Abbreviations | Definition from UMLS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All semantic groups, with LRABR abbreviations and definitions
  python umls_entity_labels_export.py MRCONSO.RRF \\
    --out output/labels.csv \\
    --mrsty MRSTY.RRF --mrdef MRDEF.RRF --lrabr LRABR \\
    --semantic-group ANATOMY,CHEM,DEVICE,DISO,FINDING,INJURY_POISONING,LABPROC,PHYS \\
    --sab ALL

  # Diseases only (MeSH category filter)
  python umls_entity_labels_export.py MRCONSO.RRF \\
    --out output/disease_labels.csv \\
    --mrhier MRHIER.RRF --mrdef MRDEF.RRF \\
    --mesh-category Diseases --sab ALL
        """,
    )

    parser.add_argument("mrconso", help="Path to MRCONSO.RRF (optionally .gz)")
    parser.add_argument("-o", "--out", required=True, help="Output CSV path")

    # Optional source files
    parser.add_argument("--mrhier", help="MRHIER.RRF — required for --mesh-category")
    parser.add_argument("--mrsty",  help="MRSTY.RRF  — required for --semantic-group")
    parser.add_argument("--mrdef",  help="MRDEF.RRF  — for definitions column")
    parser.add_argument("--lrabr",  help="SPECIALIST Lexicon LRABR — for abbreviations column")

    # Filters
    parser.add_argument("--mesh-category",
                        help="Comma-separated MeSH top-level categories (e.g. Diseases,Chemicals)")
    parser.add_argument("--semantic-group",
                        help="Comma-separated semantic groups: "
                             "ANATOMY,CHEM,DEVICE,DISO,FINDING,INJURY_POISONING,LABPROC,PHYS")
    parser.add_argument("--lat", default="ENG",
                        help="Language filter (default: ENG). Use ALL for no filter.")
    parser.add_argument("--sab", default="MSH",
                        help="Comma-separated SABs (default: MSH). Use ALL for no filter.")
    parser.add_argument("--include-suppressed", action="store_true",
                        help="Include suppressed/obsolete entries")

    args = parser.parse_args()

    # --- Validate paths ---
    for label, path in [
        ("MRCONSO", args.mrconso),
        ("MRHIER",  args.mrhier),
        ("MRSTY",   args.mrsty),
        ("MRDEF",   args.mrdef),
        ("LRABR",   args.lrabr),
    ]:
        if path and not os.path.exists(path):
            parser.error(f"{label} file not found: {path}")

    if args.mesh_category and not args.mrhier:
        parser.error("--mrhier is required with --mesh-category")
    if args.semantic_group and not args.mrsty:
        parser.error("--mrsty is required with --semantic-group")

    # --- Filters ---
    lat_filter  = None if args.lat.upper() == "ALL" else args.lat.upper()
    allowed_sab = None if (not args.sab or args.sab.upper() == "ALL") \
                       else {s.strip() for s in args.sab.split(",")}

    # --- CUI whitelists ---
    allowed_cuis: Optional[Set[str]] = None

    if args.mesh_category:
        cats = [c.strip() for c in args.mesh_category.split(",") if c.strip()]
        print(f"Loading MeSH category CUIs: {cats}")
        allowed_cuis = load_mesh_cuis(args.mrhier, cats)
        print(f"  → {len(allowed_cuis):,} CUIs")

    if args.semantic_group:
        groups = [g.strip() for g in args.semantic_group.split(",") if g.strip()]
        print(f"Loading semantic group CUIs: {groups}")
        sty_cuis = load_semantic_group_cuis(args.mrsty, groups)
        print(f"  → {len(sty_cuis):,} CUIs")
        allowed_cuis = (allowed_cuis & sty_cuis) if allowed_cuis is not None else sty_cuis
        print(f"  → {len(allowed_cuis):,} CUIs after intersection")

    # --- Enrichment ---
    def_map   = load_definitions(args.mrdef) if args.mrdef else None
    lrabr_map = load_lrabr(args.lrabr)       if args.lrabr else None

    if def_map:
        print(f"Loaded definitions for {len(def_map):,} CUIs")
    if lrabr_map:
        print(f"Loaded LRABR abbreviations for {len(lrabr_map):,} expansions")

    # --- Build entity table ---
    print(f"\nProcessing MRCONSO: {args.mrconso}")
    entity_table = build_entity_table(
        mrconso_path=args.mrconso,
        lat_filter=lat_filter,
        allowed_sab=allowed_sab,
        allowed_cuis=allowed_cuis,
        include_suppressed=args.include_suppressed,
    )
    print(f"Collected {len(entity_table):,} unique CUIs")

    # --- Export ---
    export(entity_table, def_map, lrabr_map, args.out)

    # --- Abbreviation export (separate file) ---
    if lrabr_map and args.out.endswith(".csv"):
        abbr_out_path = args.out[:-4] + "_abbreviations.csv"
        export_abbreviations(entity_table, lrabr_map, abbr_out_path)


if __name__ == "__main__":
    main()

