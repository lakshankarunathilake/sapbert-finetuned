#!/usr/bin/env python3
"""
BC5CDR Dictionary vs Dataset Analysis

Analyzes the relationship between the dictionary (candidate pool) and the test dataset.
Answers:
1. How many items are in the dictionary vs dataset?
2. What's the overlap between them?
3. Are all test CUIs present in the dictionary?
4. What are the missing entities?
5. Coverage statistics

UPDATED: Now properly handles composite CUIs (multiple CUIs separated by | or concatenated)
"""

import os
import sys
from collections import defaultdict, Counter
from tqdm import tqdm
import json
import re

# Fix: Need to go up 2 directories from coverage_reports to reach project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.data_loader import DictionaryDataset, QueryDataset


def parse_composite_cui(cui_string):
    """
    Parse composite CUIs that may be:
    1. Pipe-separated: D001234|D005678
    2. Concatenated: D001234D005678
    3. Single: D001234

    Returns: list of individual CUIs
    """
    # First check for pipe separator
    if '|' in cui_string:
        return cui_string.split('|')

    # Pattern for standard CUI format (C/D followed by 6 digits)
    cui_pattern = r'[CD]\d{6}'
    cuis = re.findall(cui_pattern, cui_string)

    # If we found multiple CUIs, it's composite
    if len(cuis) > 1:
        return cuis
    elif len(cuis) == 1:
        return cuis
    else:
        # Return as-is if no pattern match
        return [cui_string]


def check_cui_match(predicted_cui, golden_cui_string):
    """
    Check if predicted CUI matches any CUI in the golden set.
    Handles composite CUIs properly.

    Args:
        predicted_cui: Single CUI from prediction
        golden_cui_string: May contain multiple CUIs (pipe-separated or concatenated)

    Returns:
        bool: True if predicted matches any golden CUI
    """
    golden_cuis = parse_composite_cui(golden_cui_string)
    return predicted_cui in golden_cuis


def analyze_bc5cdr_coverage(data_dir, dict_path, dataset_name, output_dir="reports"):
    """
    Comprehensive analysis of dictionary coverage for BC5CDR dataset

    Args:
        data_dir: Path to processed test data
        dict_path: Path to test dictionary
        dataset_name: Name of the dataset (e.g., 'bc5cdr-chemical')
        output_dir: Directory to save output files (default: 'reports')
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print(f"BC5CDR DICTIONARY vs DATASET ANALYSIS: {dataset_name}")
    print("=" * 80)
    print()

    # ========================================
    # 1. Load Dictionary
    # ========================================
    print("📚 LOADING DICTIONARY...")
    print("-" * 80)
    dictionary = DictionaryDataset(dictionary_path=dict_path).data

    # Build dictionary mappings
    dict_cui_to_names = defaultdict(list)
    dict_name_to_cuis = defaultdict(list)
    dict_unique_cuis = set()
    dict_unique_names = set()

    for name, cui in dictionary:
        dict_cui_to_names[cui].append(name)
        dict_name_to_cuis[name].append(cui)
        dict_unique_cuis.add(cui)
        dict_unique_names.add(name)

    print(f"Dictionary Total Entries: {len(dictionary):,}")
    print(f"Dictionary Unique CUIs: {len(dict_unique_cuis):,}")
    print(f"Dictionary Unique Names: {len(dict_unique_names):,}")
    print(f"Average Names per CUI: {len(dictionary) / len(dict_unique_cuis):.2f}")
    print()

    # ========================================
    # 2. Load Test Dataset (with composite CUI handling)
    # ========================================
    print("📊 LOADING TEST DATASET...")
    print("-" * 80)
    test_queries = QueryDataset(
        data_dir=data_dir,
        filter_composite=False,
        filter_duplicate=False
    ).data

    # Build dataset mappings
    dataset_mentions = []
    dataset_unique_cuis_raw = set()  # Raw CUI strings (may be composite)
    dataset_unique_cuis_expanded = set()  # Individual CUIs expanded from composites
    dataset_unique_mentions = set()
    dataset_cui_to_mentions = defaultdict(list)
    dataset_mention_to_cuis = defaultdict(list)

    composite_cui_count = 0

    for mention, cui_string in test_queries:
        dataset_mentions.append((mention, cui_string))
        dataset_unique_cuis_raw.add(cui_string)
        dataset_unique_mentions.add(mention)
        dataset_cui_to_mentions[cui_string].append(mention)
        dataset_mention_to_cuis[mention].append(cui_string)

        # Expand composite CUIs
        individual_cuis = parse_composite_cui(cui_string)
        if len(individual_cuis) > 1:
            composite_cui_count += 1
        for cui in individual_cuis:
            dataset_unique_cuis_expanded.add(cui)

    print(f"Test Dataset Total Mentions: {len(test_queries):,}")
    print(f"Test Dataset Unique CUI Strings (raw): {len(dataset_unique_cuis_raw):,}")
    print(f"Test Dataset Composite CUI Annotations: {composite_cui_count:,}")
    print(f"Test Dataset Unique Individual CUIs (expanded): {len(dataset_unique_cuis_expanded):,}")
    print(f"Test Dataset Unique Mentions: {len(dataset_unique_mentions):,}")
    print(f"Average Mentions per CUI: {len(test_queries) / len(dataset_unique_cuis_raw):.2f}")
    print()

    # ========================================
    # 3. CUI Coverage Analysis (CORRECTED)
    # ========================================
    print("🎯 CUI COVERAGE ANALYSIS (Composite-Aware)")
    print("-" * 80)

    # Check coverage using EXPANDED individual CUIs
    test_cuis_in_dict = dataset_unique_cuis_expanded & dict_unique_cuis
    test_cuis_missing = dataset_unique_cuis_expanded - dict_unique_cuis

    # Extra CUIs in dictionary not in test
    dict_only_cuis = dict_unique_cuis - dataset_unique_cuis_expanded

    print(f"Test CUIs in Dictionary: {len(test_cuis_in_dict):,} / {len(dataset_unique_cuis_expanded):,} "
          f"({100 * len(test_cuis_in_dict) / len(dataset_unique_cuis_expanded):.2f}%)")
    print(f"Test CUIs MISSING from Dictionary: {len(test_cuis_missing):,} "
          f"({100 * len(test_cuis_missing) / len(dataset_unique_cuis_expanded):.2f}%)")
    print(f"Dictionary-only CUIs (not in test): {len(dict_only_cuis):,}")
    print()

    # Show which composite annotations have missing CUIs
    composite_with_missing = []
    for cui_string in dataset_unique_cuis_raw:
        individual_cuis = parse_composite_cui(cui_string)
        if len(individual_cuis) > 1:
            missing_parts = [c for c in individual_cuis if c not in dict_unique_cuis]
            if missing_parts:
                composite_with_missing.append((cui_string, individual_cuis, missing_parts))

    if composite_with_missing:
        print(f"⚠️  COMPOSITE ANNOTATIONS WITH MISSING CUIs: {len(composite_with_missing)}")
        for i, (raw, parts, missing) in enumerate(composite_with_missing[:10], 1):
            print(f"  {i}. Composite: {raw}")
            print(f"     Parts: {parts}")
            print(f"     Missing: {missing}")
        if len(composite_with_missing) > 10:
            print(f"  ... and {len(composite_with_missing) - 10} more")
        print()

    if test_cuis_missing:
        print("⚠️  ALL MISSING CUIs (first 20):")
        missing_sorted = sorted(test_cuis_missing)
        for i, cui in enumerate(missing_sorted[:20], 1):
            # Find which mentions/composites use this CUI
            examples = []
            for cui_string in dataset_unique_cuis_raw:
                if cui in parse_composite_cui(cui_string):
                    mentions = dataset_cui_to_mentions[cui_string][:2]
                    examples.extend(mentions)
                    if len(examples) >= 2:
                        break
            print(f"  {i}. CUI: {cui}")
            print(f"     Example mentions: {examples[:2]}")
        if len(test_cuis_missing) > 20:
            print(f"  ... and {len(test_cuis_missing) - 20} more missing CUIs")
        print()

    # ========================================
    # 4. Mention-Level Coverage
    # ========================================
    print("📝 MENTION-LEVEL COVERAGE ANALYSIS")
    print("-" * 80)

    # Check if test mentions appear in dictionary (exact match)
    mentions_in_dict = dataset_unique_mentions & dict_unique_names
    mentions_missing = dataset_unique_mentions - dict_unique_names

    print(f"Test Mentions in Dictionary (exact match): {len(mentions_in_dict):,} / {len(dataset_unique_mentions):,} "
          f"({100 * len(mentions_in_dict) / len(dataset_unique_mentions):.2f}%)")
    print(f"Test Mentions MISSING (exact match): {len(mentions_missing):,} "
          f"({100 * len(mentions_missing) / len(dataset_unique_mentions):.2f}%)")
    print()

    if mentions_missing:
        print("⚠️  MISSING MENTIONS (first 20):")
        for i, mention in enumerate(list(mentions_missing)[:20], 1):
            cuis = dataset_mention_to_cuis[mention]
            print(f"  {i}. Mention: '{mention}' -> CUI: {cuis[0]}")
        if len(mentions_missing) > 20:
            print(f"  ... and {len(mentions_missing) - 20} more missing mentions")
        print()

    # ========================================
    # 5. Query-Level Analysis (CORRECTED for Composite CUIs)
    # ========================================
    print("🔍 QUERY-LEVEL ANALYSIS")
    print("-" * 80)

    # How many queries have the correct CUI in dictionary?
    queries_with_cui_in_dict = 0
    queries_with_exact_match = 0
    queries_impossible = 0  # NO individual CUI from composite is in dictionary
    queries_partial_composite = 0  # Some but not all CUIs from composite are in dict

    for mention, cui_string in test_queries:
        individual_cuis = parse_composite_cui(cui_string)

        # Check how many individual CUIs are in dictionary
        cuis_in_dict = [cui for cui in individual_cuis if cui in dict_unique_cuis]

        if len(cuis_in_dict) > 0:
            # At least one CUI is in dictionary - query is possible
            queries_with_cui_in_dict += 1

            # Check for partial composite matches
            if len(individual_cuis) > 1 and len(cuis_in_dict) < len(individual_cuis):
                queries_partial_composite += 1

            # Check if mention exactly matches any alias of any CUI in dictionary
            for cui in cuis_in_dict:
                if mention in dict_cui_to_names[cui]:
                    queries_with_exact_match += 1
                    break
        else:
            # None of the CUIs are in dictionary - impossible
            queries_impossible += 1

    total_queries = len(test_queries)

    print(f"Queries where ANY CUI is in dictionary: {queries_with_cui_in_dict:,} / {total_queries:,} "
          f"({100 * queries_with_cui_in_dict / total_queries:.2f}%)")
    print(f"Queries with EXACT mention match in dictionary: {queries_with_exact_match:,} / {total_queries:,} "
          f"({100 * queries_with_exact_match / total_queries:.2f}%)")
    if composite_cui_count > 0:
        print(f"Composite queries with partial CUI coverage: {queries_partial_composite:,} "
              f"({100 * queries_partial_composite / total_queries:.2f}%)")
    print(f"IMPOSSIBLE queries (NO CUI in dictionary): {queries_impossible:,} / {total_queries:,} "
          f"({100 * queries_impossible / total_queries:.2f}%)")
    print()

    print("💡 THEORETICAL MAXIMUM ACCURACY:")
    print(f"   Best possible acc@1 if using this dictionary: "
          f"{100 * queries_with_cui_in_dict / total_queries:.2f}%")
    print(f"   (Assuming perfect ranking when ANY CUI exists)")
    print()

    # ========================================
    # 6. Synonym/Alias Analysis
    # ========================================
    print("🔤 SYNONYM ANALYSIS")
    print("-" * 80)

    # For CUIs that ARE in dictionary, how many aliases do they have?
    aliases_per_cui = []
    for cui in test_cuis_in_dict:
        num_aliases = len(dict_cui_to_names[cui])
        aliases_per_cui.append(num_aliases)

    if aliases_per_cui:
        print(f"Average aliases per test CUI in dictionary: {sum(aliases_per_cui) / len(aliases_per_cui):.2f}")
        print(f"Min aliases: {min(aliases_per_cui)}")
        print(f"Max aliases: {max(aliases_per_cui)}")
        print()

        # Show distribution
        alias_counts = Counter(aliases_per_cui)
        print("Alias distribution (top 10):")
        for num_aliases, count in alias_counts.most_common(10):
            print(f"  {num_aliases} aliases: {count} CUIs")
        print()

    # ========================================
    # 7. Sample Comparison
    # ========================================
    print("📋 SAMPLE COMPARISON (first 10 test queries)")
    print("-" * 80)

    for i, (mention, cui) in enumerate(test_queries[:10], 1):
        print(f"\n{i}. Test Query:")
        print(f"   Mention: '{mention}'")
        print(f"   Golden CUI: {cui}")

        if cui in dict_unique_cuis:
            dict_names_for_cui = dict_cui_to_names[cui][:5]  # Show first 5 aliases
            print(f"   ✅ CUI in dictionary with {len(dict_cui_to_names[cui])} aliases:")
            for alias in dict_names_for_cui:
                match = "🎯 EXACT" if alias == mention else "   "
                print(f"      {match} '{alias}'")
            if len(dict_cui_to_names[cui]) > 5:
                print(f"      ... and {len(dict_cui_to_names[cui]) - 5} more aliases")
        else:
            print(f"   ❌ CUI NOT in dictionary (impossible to get correct)")

    print()

    # ========================================
    # 8. Summary Statistics
    # ========================================
    print("=" * 80)
    print("📊 SUMMARY STATISTICS")
    print("=" * 80)

    summary = {
        "dataset_name": dataset_name,
        "dictionary": {
            "total_entries": len(dictionary),
            "unique_cuis": len(dict_unique_cuis),
            "unique_names": len(dict_unique_names),
            "avg_names_per_cui": len(dictionary) / len(dict_unique_cuis) if dict_unique_cuis else 0
        },
        "test_dataset": {
            "total_mentions": len(test_queries),
            "unique_cuis_raw": len(dataset_unique_cuis_raw),
            "unique_cuis_expanded": len(dataset_unique_cuis_expanded),
            "composite_cui_count": composite_cui_count,
            "unique_mentions": len(dataset_unique_mentions),
            "avg_mentions_per_cui": len(test_queries) / len(dataset_unique_cuis_raw) if dataset_unique_cuis_raw else 0
        },
        "coverage": {
            "cui_coverage_pct": 100 * len(test_cuis_in_dict) / len(dataset_unique_cuis_expanded) if dataset_unique_cuis_expanded else 0,
            "mention_coverage_pct": 100 * len(mentions_in_dict) / len(dataset_unique_mentions) if dataset_unique_mentions else 0,
            "queries_possible_pct": 100 * queries_with_cui_in_dict / total_queries if total_queries else 0,
            "queries_with_exact_match_pct": 100 * queries_with_exact_match / total_queries if total_queries else 0,
            "theoretical_max_accuracy_pct": 100 * queries_with_cui_in_dict / total_queries if total_queries else 0
        },
        "missing": {
            "missing_cuis": len(test_cuis_missing),
            "missing_mentions": len(mentions_missing),
            "impossible_queries": queries_impossible,
            "composite_with_missing_parts": len(composite_with_missing) if composite_with_missing else 0
        }
    }

    # Print summary
    for category, stats in summary.items():
        if category == "dataset_name":
            continue
        print(f"\n{category.upper().replace('_', ' ')}:")
        for key, value in stats.items():
            key_formatted = key.replace('_', ' ').title()
            if isinstance(value, float):
                print(f"  {key_formatted}: {value:.2f}")
            else:
                print(f"  {key_formatted}: {value:,}")

    print()
    print("=" * 80)

    # Save to JSON
    output_file = os.path.join(output_dir, f"{dataset_name}_dictionary_coverage_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"📁 Detailed analysis saved to: {output_file}")
    print()

    # ========================================
    # 9. Generate Detailed Missing Data Report
    # ========================================
    print("📋 GENERATING DETAILED MISSING DATA REPORT...")
    print("-" * 80)

    # Prepare detailed missing mentions report
    missing_mentions_details = []
    for mention in sorted(mentions_missing):
        cui_strings = dataset_mention_to_cuis[mention]
        unique_cuis = list(set(cui_strings))

        # Count occurrences
        occurrence_count = sum(1 for m, c in test_queries if m == mention)

        # Check if CUI exists in dictionary (under different name)
        cui_exists_in_dict = []
        cui_missing_from_dict = []

        for cui_str in unique_cuis:
            individual_cuis = parse_composite_cui(cui_str)
            for cui in individual_cuis:
                if cui in dict_unique_cuis:
                    cui_exists_in_dict.append(cui)
                    # Get example aliases from dictionary
                    aliases = dict_cui_to_names[cui][:5]
                else:
                    cui_missing_from_dict.append(cui)

        missing_mentions_details.append({
            'mention': mention,
            'occurrence_count': occurrence_count,
            'cui_strings': unique_cuis,
            'cui_exists_in_dict': list(set(cui_exists_in_dict)),
            'cui_missing_from_dict': list(set(cui_missing_from_dict)),
            'example_aliases': [dict_cui_to_names[cui][:3] for cui in set(cui_exists_in_dict)] if cui_exists_in_dict else [],
            'status': 'CUI_EXISTS_DIFFERENT_NAME' if cui_exists_in_dict else 'CUI_MISSING'
        })

    # Prepare detailed missing CUIs report
    missing_cuis_details = []
    for cui in sorted(test_cuis_missing):
        # Find all mentions that map to this CUI
        mentions_for_cui = []
        for cui_string in dataset_unique_cuis_raw:
            if cui in parse_composite_cui(cui_string):
                mentions_for_cui.extend(dataset_cui_to_mentions[cui_string])

        unique_mentions = list(set(mentions_for_cui))
        occurrence_count = len(mentions_for_cui)

        missing_cuis_details.append({
            'cui': cui,
            'occurrence_count': occurrence_count,
            'unique_mentions': unique_mentions[:10],  # Limit to 10 examples
            'total_unique_mentions': len(unique_mentions)
        })

    # Prepare detailed impossible queries report
    impossible_queries_details = []
    for mention, cui_string in test_queries:
        individual_cuis = parse_composite_cui(cui_string)
        cuis_in_dict = [cui for cui in individual_cuis if cui in dict_unique_cuis]

        if len(cuis_in_dict) == 0:
            # This is an impossible query
            impossible_queries_details.append({
                'mention': mention,
                'golden_cui_string': cui_string,
                'individual_cuis': individual_cuis,
                'is_composite': len(individual_cuis) > 1
            })

    # Create comprehensive missing data report
    missing_data_report = {
        'dataset_name': dataset_name,
        'summary': {
            'total_missing_mentions': len(mentions_missing),
            'total_missing_cuis': len(test_cuis_missing),
            'total_impossible_queries': queries_impossible,
            'composite_with_missing_parts': len(composite_with_missing) if composite_with_missing else 0
        },
        'missing_mentions_detailed': missing_mentions_details,
        'missing_cuis_detailed': missing_cuis_details,
        'impossible_queries_detailed': impossible_queries_details[:100],  # Limit to first 100
        'composite_with_missing_parts_detailed': [
            {
                'composite_cui_string': raw,
                'individual_cuis': parts,
                'missing_cuis': missing,
                'example_mentions': dataset_cui_to_mentions[raw][:5]
            }
            for raw, parts, missing in composite_with_missing
        ]
    }

    # Save detailed missing data report
    missing_report_file = os.path.join(output_dir, f"{dataset_name}_missing_data_detailed.json")
    with open(missing_report_file, 'w') as f:
        json.dump(missing_data_report, f, indent=2, ensure_ascii=False)
    print(f"📁 Detailed missing data report saved to: {missing_report_file}")

    # Also generate human-readable text report
    text_report_file = os.path.join(output_dir, f"{dataset_name}_missing_data_detailed.txt")
    with open(text_report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"DETAILED MISSING DATA REPORT: {dataset_name}\n")
        f.write("="*80 + "\n\n")

        # Summary
        f.write("SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Missing Mentions: {len(mentions_missing)}\n")
        f.write(f"Total Missing CUIs: {len(test_cuis_missing)}\n")
        f.write(f"Total Impossible Queries: {queries_impossible}\n")
        f.write(f"Composite with Missing Parts: {len(composite_with_missing) if composite_with_missing else 0}\n\n")

        # Missing Mentions Section
        f.write("\n" + "="*80 + "\n")
        f.write("MISSING MENTIONS DETAILED\n")
        f.write("="*80 + "\n")
        f.write(f"Total: {len(mentions_missing)} unique mention strings not found in dictionary\n\n")

        # Group by status
        cui_exists = [m for m in missing_mentions_details if m['status'] == 'CUI_EXISTS_DIFFERENT_NAME']
        cui_missing = [m for m in missing_mentions_details if m['status'] == 'CUI_MISSING']

        f.write(f"\n{'='*80}\n")
        f.write(f"A. MENTIONS WHERE CUI EXISTS (but under different name)\n")
        f.write(f"{'='*80}\n")
        f.write(f"Count: {len(cui_exists)}\n")
        f.write("These mentions could potentially be linked via semantic similarity.\n\n")

        for i, item in enumerate(cui_exists[:50], 1):  # First 50
            f.write(f"{i}. Mention: '{item['mention']}'\n")
            f.write(f"   Occurrences: {item['occurrence_count']}\n")
            f.write(f"   CUI(s): {', '.join(item['cui_strings'])}\n")
            f.write(f"   ✅ CUI exists in dictionary as:\n")
            for cui_idx, (cui, aliases) in enumerate(zip(item['cui_exists_in_dict'], item['example_aliases'])):
                f.write(f"      {cui}: {', '.join(aliases[:3])}\n")
            f.write("\n")

        if len(cui_exists) > 50:
            f.write(f"... and {len(cui_exists) - 50} more mentions where CUI exists\n\n")

        f.write(f"\n{'='*80}\n")
        f.write(f"B. MENTIONS WHERE CUI IS COMPLETELY MISSING\n")
        f.write(f"{'='*80}\n")
        f.write(f"Count: {len(cui_missing)}\n")
        f.write("These are IMPOSSIBLE queries - cannot be answered with this dictionary.\n\n")

        for i, item in enumerate(cui_missing[:50], 1):  # First 50
            f.write(f"{i}. Mention: '{item['mention']}'\n")
            f.write(f"   Occurrences: {item['occurrence_count']}\n")
            f.write(f"   CUI(s): {', '.join(item['cui_strings'])}\n")
            f.write(f"   ❌ Missing CUIs: {', '.join(item['cui_missing_from_dict'])}\n")
            f.write("\n")

        if len(cui_missing) > 50:
            f.write(f"... and {len(cui_missing) - 50} more mentions with missing CUIs\n\n")

        # Missing CUIs Section
        f.write("\n" + "="*80 + "\n")
        f.write("MISSING CUIs DETAILED\n")
        f.write("="*80 + "\n")
        f.write(f"Total: {len(test_cuis_missing)} unique CUIs not found in dictionary\n\n")

        # Sort by occurrence count
        missing_cuis_sorted = sorted(missing_cuis_details, key=lambda x: x['occurrence_count'], reverse=True)

        for i, item in enumerate(missing_cuis_sorted[:50], 1):  # First 50
            f.write(f"{i}. CUI: {item['cui']}\n")
            f.write(f"   Occurrences in test set: {item['occurrence_count']}\n")
            f.write(f"   Unique mentions: {item['total_unique_mentions']}\n")
            f.write(f"   Example mentions:\n")
            for mention in item['unique_mentions'][:5]:
                f.write(f"      - '{mention}'\n")
            f.write("\n")

        if len(missing_cuis_sorted) > 50:
            f.write(f"... and {len(missing_cuis_sorted) - 50} more missing CUIs\n\n")

        # Impossible Queries Section
        if impossible_queries_details:
            f.write("\n" + "="*80 + "\n")
            f.write("IMPOSSIBLE QUERIES DETAILED\n")
            f.write("="*80 + "\n")
            f.write(f"Total: {len(impossible_queries_details)} queries that cannot be answered\n\n")

            for i, item in enumerate(impossible_queries_details[:100], 1):  # First 100
                f.write(f"{i}. Mention: '{item['mention']}'\n")
                f.write(f"   Golden CUI: {item['golden_cui_string']}\n")
                if item['is_composite']:
                    f.write(f"   Type: COMPOSITE ({len(item['individual_cuis'])} CUIs)\n")
                    f.write(f"   Individual CUIs: {', '.join(item['individual_cuis'])}\n")
                else:
                    f.write(f"   Type: SINGLE\n")
                f.write(f"   Status: ❌ IMPOSSIBLE (no CUI in dictionary)\n")
                f.write("\n")

            if len(impossible_queries_details) > 100:
                f.write(f"... and {len(impossible_queries_details) - 100} more impossible queries\n\n")

    print(f"📁 Human-readable missing data report saved to: {text_report_file}")
    print()

    return summary


def main():
    """Run analysis for BC5CDR datasets"""

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to evaluation directory
    evaluation_dir = os.path.dirname(script_dir)

    datasets = [
        {
            "name": "bc5cdr-chemical",
            "dict_path": os.path.join(evaluation_dir, "data/bc5cdr-chemical/test_dictionary.txt"),
            "data_dir": os.path.join(evaluation_dir, "data/bc5cdr-chemical/processed_test")
        },
        {
            "name": "bc5cdr-disease",
            "dict_path": os.path.join(evaluation_dir, "data/bc5cdr-disease/test_dictionary.txt"),
            "data_dir": os.path.join(evaluation_dir, "data/bc5cdr-disease/processed_test")
        }
    ]

    # Check which datasets are available
    available_datasets = []
    for dataset in datasets:
        if os.path.exists(dataset["dict_path"]) and os.path.exists(dataset["data_dir"]):
            available_datasets.append(dataset)

    if not available_datasets:
        print("❌ No BC5CDR datasets found!")
        print("Expected paths:")
        for dataset in datasets:
            print(f"  - {dataset['dict_path']}")
            print(f"  - {dataset['data_dir']}")
        return

    # Analyze each available dataset
    all_summaries = {}
    for dataset in available_datasets:
        summary = analyze_bc5cdr_coverage(
            data_dir=dataset["data_dir"],
            dict_path=dataset["dict_path"],
            dataset_name=dataset["name"]
        )
        all_summaries[dataset["name"]] = summary
        print("\n" * 2)

    # Comparison if both datasets available
    if len(available_datasets) == 2:
        print("=" * 80)
        print("📊 CHEMICAL vs DISEASE COMPARISON")
        print("=" * 80)

        for key in ["dictionary", "test_dataset", "coverage"]:
            print(f"\n{key.upper().replace('_', ' ')}:")
            print(f"{'Metric':<40} {'Chemical':>15} {'Disease':>15}")
            print("-" * 72)

            for metric in all_summaries["bc5cdr-chemical"][key].keys():
                chem_val = all_summaries["bc5cdr-chemical"][key][metric]
                dis_val = all_summaries["bc5cdr-disease"][key][metric]

                metric_name = metric.replace('_', ' ').title()

                if isinstance(chem_val, float):
                    print(f"{metric_name:<40} {chem_val:>15.2f} {dis_val:>15.2f}")
                else:
                    print(f"{metric_name:<40} {chem_val:>15,} {dis_val:>15,}")


if __name__ == "__main__":
    main()
