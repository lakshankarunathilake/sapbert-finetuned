#!/usr/bin/env python3
"""
Optimized UMLS Synonym Exporter

Reads UMLS Metathesaurus MRCONSO.RRF and writes CSV with columns:
CUI,term1,term2,...

Key optimizations:
- Streaming CSV write (no intermediate DataFrame)
- Memory-efficient processing with generators
- Faster string normalization
- Optimized deduplication using sets
- Better progress tracking
- Configurable batch processing

Default behavior:
- English terms only (LAT == "ENG")
- Excludes suppressed/obsolete entries (SUPPRESS in {"O", "Y"})
- Deduplicates synonyms (case- and whitespace-insensitive)
- Sorts synonyms case-insensitively for stable output
"""

import argparse
import csv
import gzip
import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Generator
from tqdm import tqdm


def get_file_lines(path: str) -> int:
    """Get approximate line count for progress bar."""
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
    """Open file with appropriate handler for .gz or regular files."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore", newline="")
    return open(path, "r", encoding="utf-8", errors="ignore", newline="")


def normalize_term(term: str) -> str:
    """Fast string normalization for deduplication."""
    return ' '.join(term.split()).strip().lower()


def parse_mrconso_line(line: str) -> Optional[Dict[str, str]]:
    """Parse a single MRCONSO line into relevant fields."""
    parts = line.rstrip('\n').split('|')
    if len(parts) < 17:
        return None

    return {
        'CUI': parts[0],
        'LAT': parts[1],
        'ISPREF': parts[6],
        'SAB': parts[11],
        'TTY': parts[12],
        'CODE': parts[13],  # Source-specific identifier (e.g., MeSH ID)
        'STR': parts[14].strip(),
        'SUPPRESS': parts[16] if len(parts) > 16 else ''
    }


def should_include_term(record: Dict[str, str],
                        lat_filter: Optional[str],
                        allowed_sab: Optional[Set[str]],
                        allowed_tty: Optional[Set[str]],
                        include_suppressed: bool) -> bool:
    """Check if a term should be included based on filters."""
    if lat_filter and record['LAT'] != lat_filter:
        return False
    if not include_suppressed and record['SUPPRESS'] in {'O', 'Y'}:
        return False
    if allowed_sab and record['SAB'] not in allowed_sab:
        return False
    if allowed_tty and record['TTY'] not in allowed_tty:
        return False
    return True


def process_cui_terms(terms: List[Tuple[str, bool]],
                      prefer_preferred_first: bool,
                      min_terms: int) -> Optional[List[str]]:
    """Process and deduplicate terms for a CUI."""
    if not terms:
        return None

    # Deduplicate using normalized terms as keys
    seen: Dict[str, Tuple[str, bool]] = {}

    for original_term, is_preferred in terms:
        normalized = normalize_term(original_term)
        if normalized and normalized not in seen:
            seen[normalized] = (original_term, is_preferred)

    if len(seen) < min_terms:
        return None

    # Sort terms
    unique_terms = list(seen.values())
    if prefer_preferred_first:
        # Sort by: preferred first, then alphabetically
        unique_terms.sort(key=lambda x: (not x[1], x[0].lower()))
    else:
        # Sort alphabetically only
        unique_terms.sort(key=lambda x: x[0].lower())

    return [term[0] for term in unique_terms]


def process_mrconso_streaming(file_path: str,
                              lat_filter: Optional[str],
                              allowed_sab: Optional[Set[str]],
                              allowed_tty: Optional[Set[str]],
                              include_suppressed: bool,
                              prefer_preferred_first: bool,
                              min_terms: int,
                              max_rows: int,
                              use_source_id: bool = False) -> Generator[List[str], None, None]:
    """Stream process MRCONSO file and yield processed rows."""

    # Use source CODE instead of CUI as the grouping key when use_source_id=True
    by_identifier: Dict[str, List[Tuple[str, bool]]] = defaultdict(list)
    total_lines = get_file_lines(file_path)
    rows_written = 0

    with open_file(file_path) as f:
        # First pass: collect all terms by identifier (CUI or source CODE)
        for line in tqdm(f, total=total_lines, desc="Reading MRCONSO", unit="lines"):
            record = parse_mrconso_line(line)
            if not record:
                continue

            if not should_include_term(record, lat_filter, allowed_sab, allowed_tty, include_suppressed):
                continue

            # Use either source CODE or CUI as identifier
            identifier = record['CODE'] if use_source_id else record['CUI']
            if not identifier:  # Skip if no identifier available
                continue

            is_preferred = (record['ISPREF'] == 'Y') or (record['TTY'] in {'PT', 'PN'})
            by_identifier[identifier].append((record['STR'], is_preferred))

    # Second pass: process and yield results
    for identifier in tqdm(by_identifier.keys(), desc="Processing identifiers", unit="id"):
        processed_terms = process_cui_terms(
            by_identifier[identifier], prefer_preferred_first, min_terms
        )

        if processed_terms:
            yield [identifier] + processed_terms
            rows_written += 1
            if max_rows and rows_written >= max_rows:
                break


def write_csv_streaming(output_path: str,
                        rows_generator: Generator[List[str], None, None],
                        use_source_id: bool = False) -> None:
    """Write CSV with streaming approach to minimize memory usage."""

    # Create headers - adjust based on identifier type
    id_column = 'SourceID' if use_source_id else 'CUI'
    headers = [id_column, 'Terms']

    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for row in tqdm(rows_generator, desc="Writing CSV", unit="rows"):
            if len(row) >= 2:
                # Join all terms (excluding identifier) with ||
                identifier = row[0]
                terms = '||'.join(row[1:])
                writer.writerow([identifier, terms])


def main():
    parser = argparse.ArgumentParser(
        description="Export UMLS CUI → synonyms from MRCONSO.RRF (Optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("mrconso", help="Path to MRCONSO.RRF (optionally .gz)")
    parser.add_argument("-o", "--out", required=True, help="Output CSV file path")
    parser.add_argument("--lat", default="ENG",
                        help="Language filter (default: ENG). Use 'ALL' for no filter.")
    parser.add_argument("--sab",
                        help="Comma-separated allowed source vocabularies (e.g., 'SNOMEDCT_US,MSH')")
    parser.add_argument("--tty",
                        help="Comma-separated allowed term types (e.g., 'PT,SY,PN')")
    parser.add_argument("--include-suppressed", action="store_true",
                        help="Include suppressed/obsolete entries (default: exclude)")
    parser.add_argument("--min-terms", type=int, default=1,
                        help="Only output CUIs with at least this many terms (default: 1)")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="Stop after writing N CUIs (for testing). 0 = no limit")
    parser.add_argument("--prefer-preferred-first", action="store_true",
                        help="Place preferred terms (ISPREF=Y or TTY in {PT,PN}) first")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output to show SAB values found")
    parser.add_argument("--use-source-id", action="store_true",
                        help="Use source vocabulary ID (CODE field) instead of UMLS CUI")

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.mrconso):
        parser.error(f"Input file not found: {args.mrconso}")

    # Parse filters
    allowed_sab = set(s.strip() for s in args.sab.split(',')) if args.sab else None
    allowed_tty = set(t.strip() for t in args.tty.split(',')) if args.tty else None
    lat_filter = None if args.lat.upper() == 'ALL' else args.lat.upper()

    print(f"Processing {args.mrconso}...")
    print(f"Language filter: {lat_filter or 'ALL'}")
    print(f"Source vocabularies: {args.sab or 'ALL'}")
    print(f"Term types: {args.tty or 'ALL'}")
    print(f"Include suppressed: {args.include_suppressed}")
    print(f"Min terms per CUI: {args.min_terms}")
    print(f"Max rows: {args.max_rows or 'unlimited'}")
    print(f"Use source ID: {args.use_source_id}")

    # Debug mode - show available SAB values
    if args.debug and args.sab:
        print("\nDEBUG: Checking available SAB values...")
        sab_counts = {}
        with open_file(args.mrconso) as f:
            for i, line in enumerate(f):
                if i >= 10000:  # Check first 10k lines
                    break
                record = parse_mrconso_line(line)
                if record:
                    sab = record['SAB']
                    sab_counts[sab] = sab_counts.get(sab, 0) + 1

        print(f"Found SAB values in first 10k lines: {sorted(sab_counts.keys())}")
        if 'MESH' in sab_counts:
            print(f"MESH found: {sab_counts['MESH']} occurrences")
        if 'MSH' in sab_counts:
            print(f"MSH found: {sab_counts['MSH']} occurrences")
        print()

    # Process and write
    rows_gen = process_mrconso_streaming(
        args.mrconso,
        lat_filter,
        allowed_sab,
        allowed_tty,
        args.include_suppressed,
        args.prefer_preferred_first,
        args.min_terms,
        args.max_rows,
        args.use_source_id
    )

    write_csv_streaming(args.out, rows_gen, args.use_source_id)
    print(f"Output written to: {args.out}")


if __name__ == "__main__":
    main()