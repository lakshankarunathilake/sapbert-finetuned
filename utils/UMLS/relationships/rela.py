import pandas as pd
from collections import defaultdict


# MRREL.RRF file format (pipe-delimited)
# Columns: CUI1|AUI1|STYPE1|REL|CUI2|AUI2|STYPE2|RELA|RUI|SRUI|SAB|SL|RG|DIR|SUPPRESS|CVF|
# SAB (Source) is at index 10

def get_available_sources(filepath, sample_size=1000000):
    """
    Get list of all available sources in the file
    """
    sources = set()

    print("Scanning file for available sources...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num > sample_size:
                break

            fields = line.strip().split('|')
            if len(fields) > 10:
                source = fields[10].strip()
                if source:
                    sources.add(source)

    return sorted(sources)


def extract_unique_rela(filepath, filter_sources=None):
    """
    Extract unique RELA values from MRREL.RRF file
    RELA is in column 8 (index 7), REL is in column 4 (index 3), SAB is in column 11 (index 10)

    Args:
        filepath: Path to MRREL.RRF file
        filter_sources: List of source abbreviations to include (e.g., ['MSH', 'SNOMEDCT_US'])
                       If None, includes all sources
    """
    unique_rela = set()

    print("Reading MRREL.RRF file...")
    if filter_sources:
        print(f"Filtering by sources: {', '.join(filter_sources)}")

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 1000000 == 0:
                print(f"Processed {line_num:,} lines...")

            fields = line.strip().split('|')

            # REL is at index 3, RELA is at index 7, SAB is at index 10
            if len(fields) > 10:
                rel = fields[3].strip()
                rela = fields[7].strip()
                source = fields[10].strip()

                # Apply source filter if specified
                if filter_sources and source not in filter_sources:
                    continue

                if rela:  # Only add non-empty RELA values
                    unique_rela.add((rel, rela, source))

    return sorted(unique_rela)


def extract_with_counts(filepath, filter_sources=None):
    """
    Extract unique RELA values with their counts, including REL and source

    Args:
        filepath: Path to MRREL.RRF file
        filter_sources: List of source abbreviations to include
    """
    rela_counts = {}
    source_stats = defaultdict(int)

    print("Reading MRREL.RRF file and counting RELA values...")
    if filter_sources:
        print(f"Filtering by sources: {', '.join(filter_sources)}")

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 1000000 == 0:
                print(f"Processed {line_num:,} lines...")

            fields = line.strip().split('|')

            if len(fields) > 10:
                rel = fields[3].strip()
                rela = fields[7].strip()
                source = fields[10].strip()

                # Apply source filter if specified
                if filter_sources and source not in filter_sources:
                    continue

                if rela:
                    key = (rel, rela, source)
                    rela_counts[key] = rela_counts.get(key, 0) + 1
                    source_stats[source] += 1

    # Sort by count (descending)
    sorted_rela = sorted(rela_counts.items(), key=lambda x: x[1], reverse=True)

    # Print source statistics
    print("\n" + "=" * 60)
    print("SOURCE STATISTICS")
    print("=" * 60)
    print(f"{'Source':<20} {'Count':>15}")
    print("-" * 35)
    for source, count in sorted(source_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"{source:<20} {count:>15,}")

    return sorted_rela


def extract_by_source_separate(filepath, sources_of_interest):
    """
    Extract relationships for each source separately
    """
    results = {}

    for source in sources_of_interest:
        print(f"\nProcessing source: {source}")
        rela_counts = extract_with_counts(filepath, filter_sources=[source])
        results[source] = rela_counts

    return results


def extract_rela_counts_aggregated(filepath, filter_sources=None):
    """
    Aggregate counts by RELA (and REL) across all sources, ignoring the source.
    Args:
        filepath: Path to MRREL.RRF file
        filter_sources: List of source abbreviations to include (optional)
    Returns:
        Dictionary with (REL, RELA) as key and count as value
    """
    rela_counts = {}
    print("Reading MRREL.RRF file and aggregating RELA counts (ignoring source)...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 1000000 == 0:
                print(f"Processed {line_num:,} lines...")
            fields = line.strip().split('|')
            if len(fields) > 10:
                rel = fields[3].strip()
                rela = fields[7].strip()
                source = fields[10].strip()
                if filter_sources and source not in filter_sources:
                    continue
                if rela:
                    key = (rel, rela)
                    rela_counts[key] = rela_counts.get(key, 0) + 1
    # Sort by count (descending)
    sorted_rela = sorted(rela_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_rela


# Main execution
if __name__ == "__main__":
    filepath = "/Users/lakshankarunathilake/Documents/UMLS dataset/2025AA 2/META/MRREL.RRF"

    # === CONFIGURATION FLAG ===
    GROUP_BY_SOURCE = False  # Set to True to create separate files per source, False for a single file
    # =========================

    # First, discover available sources
    print("\n" + "=" * 80)
    print("DISCOVERING AVAILABLE SOURCES")
    print("=" * 80)
    available_sources = get_available_sources(filepath)
    print(f"\nFound {len(available_sources)} sources:")
    for source in available_sources:
        print(f"  - {source}")

    # Save available sources
    with open('available_sources.txt', 'w') as f:
        f.write("Available Sources in UMLS\n")
        f.write("=" * 60 + "\n\n")
        for source in available_sources:
            f.write(f"{source}\n")
    print("\nSaved to 'available_sources.txt'")

    # Define sources of interest for medical/clinical relationships
    # Common important sources:
    CLINICAL_SOURCES = [
        # 'MSH',  # MeSH (Medical Subject Headings)
        # 'SNOMEDCT_US',  # SNOMED CT
        # 'RXNORM',  # RxNorm (drugs)
        # 'NCI',  # NCI Thesaurus
        # 'MEDLINEPLUS',  # MedlinePlus
        # 'NDFRT',  # NDF-RT (drug classification)
    ]

    # Filter to only sources that exist in the file
    if not CLINICAL_SOURCES:
        filter_sources = available_sources  # Do not filter if empty
    else:
        filter_sources = [s for s in CLINICAL_SOURCES if s in available_sources]

    print("\n" + "=" * 80)
    print("FILTERING BY CLINICAL SOURCES")
    print("=" * 80)
    print(f"Using sources: {', '.join(filter_sources)}")

    # Option 1: Get unique RELA values with source filtering
    print("\n" + "=" * 80)
    print("Extracting unique RELA values (filtered by source)...")
    print("=" * 80)
    unique_relas = extract_unique_rela(filepath, filter_sources=filter_sources)

    print(f"\nTotal unique REL-RELA-SOURCE combinations: {len(unique_relas)}")
    print("\nUnique REL, RELA and SOURCE values:")
    print("-" * 80)
    print(f"{'REL':<10} {'RELA':<40} {'Source':<20}")
    print("-" * 80)
    for rel, rela, source in unique_relas[:50]:  # Show first 50
        print(f"{rel:<10} {rela:<40} {source:<20}")

    # Save to file
    with open('unique_rela_types_filtered.txt', 'w') as f:
        f.write(f"Filtered by sources: {', '.join(filter_sources)}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'REL':<10} {'RELA':<40} {'Source':<20}\n")
        f.write("-" * 80 + "\n")
        for rel, rela, source in unique_relas:
            f.write(f"{rel:<10} {rela:<40} {source:<20}\n")
    print(f"\nSaved to 'unique_rela_types_filtered.txt'")

    # Option 2: Get counts with source filtering
    print("\n" + "=" * 80)
    print("Getting RELA counts (filtered by source)...")
    print("=" * 80)
    rela_with_counts = extract_with_counts(filepath, filter_sources=filter_sources)

    print(f"\nREL-RELA-SOURCE types with counts (sorted by frequency):")
    print("-" * 95)
    print(f"{'REL':<10} {'RELA':<40} {'Source':<20} {'Count':>15}")
    print("-" * 95)
    for (rel, rela, source), count in rela_with_counts[:50]:  # Show top 50
        print(f"{rel:<10} {rela:<40} {source:<20} {count:>15,}")

    # Save counts to file
    with open('rela_types_with_counts_filtered.txt', 'w') as f:
        f.write(f"Filtered by sources: {', '.join(filter_sources)}\n")
        f.write("=" * 95 + "\n\n")
        f.write(f"{'REL':<10} {'RELA':<40} {'Source':<20} {'Count':>15}\n")
        f.write("-" * 95 + "\n")
        for (rel, rela, source), count in rela_with_counts:
            f.write(f"{rel:<10} {rela:<40} {source:<20} {count:>15,}\n")
    print(f"\nSaved to 'rela_types_with_counts_filtered.txt'")

    # Option 3: Extract relationships by each source separately or all together
    print("\n" + "=" * 80)
    print("Extracting relationships by source...")
    print("=" * 80)

    if not GROUP_BY_SOURCE:
        # Create a single file for all sources (no grouping, aggregate by RELA/REL)
        print("Grouping disabled. Creating a single file for all sources (aggregated by RELA/REL)...")
        all_rela = extract_rela_counts_aggregated(filepath, filter_sources=filter_sources if filter_sources else None)
        filename = 'rela_types_all_sources.txt'
        with open(filename, 'w') as f:
            f.write("ALL SOURCES (Aggregated by RELA/REL, ignoring source)\n")
            f.write("=" * 65 + "\n\n")
            f.write(f"{'REL':<10} {'RELA':<40} {'Count':>10}\n")
            f.write("-" * 65 + "\n")
            for (rel, rela), count in all_rela:
                f.write(f"{rel:<10} {rela:<40} {count:>10,}\n")
        print(f"  Saved to '{filename}'")
    else:
        for source in filter_sources:
            print(f"\nProcessing {source}...")
            source_rela = extract_with_counts(filepath, filter_sources=[source])

            filename = f'rela_{source}.txt'
            with open(filename, 'w') as f:
                f.write(f"Source: {source}\n")
                f.write("=" * 95 + "\n\n")
                f.write(f"Total relationships: {sum(count for _, count in source_rela):,}\n")
                f.write(f"Unique REL-RELA pairs: {len(source_rela)}\n\n")
                f.write(f"{'REL':<10} {'RELA':<50} {'Count':>15}\n")
                f.write("-" * 75 + "\n")
                for (rel, rela, src), count in source_rela:
                    f.write(f"{rel:<10} {rela:<50} {count:>15,}\n")

            print(f"  Saved to '{filename}'")

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - available_sources.txt")
    print("  - unique_rela_types_filtered.txt")
    print("  - rela_types_with_counts_filtered.txt")
    if not GROUP_BY_SOURCE:
        print("  - rela_types_all_sources.txt")
    else:
        for source in filter_sources:
            print(f"  - rela_{source}.txt")
