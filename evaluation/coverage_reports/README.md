# Dictionary Coverage Analysis Tool

Comprehensive analysis tool for evaluating biomedical dictionary coverage against BC5CDR test datasets. Supports multiple dictionary formats including BC5CDR standard format and MeSH CSV format.

## Overview

This tool analyzes how well a dictionary covers entities in a test dataset by examining:
- **CUI Coverage**: Percentage of test CUIs present in the dictionary
- **Mention Coverage**: Exact string match coverage
- **Query-Level Analysis**: How many queries can theoretically be answered
- **Synonym Richness**: Average number of aliases per concept
- **Missing Entities**: Detailed analysis of gaps in coverage

## Features

✅ **Multi-Format Support**
- BC5CDR format (`.txt`): `CUI<TAB>term` or `CUI<SPACE>term`
- MeSH CSV format (`.csv`): `SourceID,Terms` with `||` separated synonyms

✅ **Auto-Detection**
- Automatically detects dictionary format based on file extension
- Smart parsing of both tab and space-separated BC5CDR files

✅ **Comprehensive Reports**
- Console output with detailed statistics
- JSON reports saved for further analysis
- Coverage percentages, missing entities, and recommendations

✅ **Composite CUI Handling**
- Properly handles multi-concept annotations (e.g., `D001234D005678`)
- Identifies partial coverage in composite annotations

## Installation

### Prerequisites

```bash
# Python 3.7+ required
pip install tqdm
```

### Directory Structure

```
evaluation/coverage_reports/
├── README.md                                    # This file
├── analyze_dictionary_coverage_flexible.py      # Main analysis script
├── run_mesh_coverage_analysis.py                # Convenience script for standard analyses
└── reports/                                     # Output directory (auto-created)
    ├── bc5cdr-chemical-mesh_coverage_analysis.json
    ├── bc5cdr-disease-mesh_coverage_analysis.json
    └── MESH_COVERAGE_SUMMARY.md
```

## Quick Start

### Option 1: Run All Standard Analyses

The easiest way to analyze MeSH and BC5CDR dictionaries:

```bash
cd evaluation/coverage_reports
python run_mesh_coverage_analysis.py
```

This will automatically:
1. Analyze MeSH Chemicals dictionary vs BC5CDR-Chemical test set
2. Analyze MeSH Diseases dictionary vs BC5CDR-Disease test set
3. Compare with original BC5CDR dictionaries (baseline)
4. Save all reports to `reports/` directory

**Expected file locations** (auto-configured):
- MeSH dictionaries: `utils/UMLS/mesh_chemicals_eng_bc5cdr_format.txt`, `mesh_diseases_eng_bc5cdr_format.txt`
- Test data: `evaluation/data/bc5cdr-chemical/processed_test`, `evaluation/data/bc5cdr-disease/processed_test`
- Original dictionaries: `evaluation/data/bc5cdr-chemical/test_dictionary.txt`, `evaluation/data/bc5cdr-disease/test_dictionary.txt`

### Option 2: Analyze a Custom Dictionary

For analyzing a specific dictionary against a test dataset:

```bash
python analyze_dictionary_coverage_flexible.py \
  --dict-path /path/to/your_dictionary.csv \
  --data-dir /path/to/test_data \
  --dataset-name my-custom-analysis \
  --mesh-separator "||"
```

## Usage Examples

### Example 1: Analyze MeSH Chemicals Dictionary

```bash
python analyze_dictionary_coverage_flexible.py \
  --dict-path ../../utils/UMLS/mesh_chemicals_eng.csv \
  --data-dir ../data/bc5cdr-chemical/processed_test \
  --dataset-name bc5cdr-chemical-mesh \
  --mesh-separator "||"
```

```bash
python analyze_dictionary_coverage_flexible.py \
  --dict-path ../../utils/UMLS/mesh_diseases_eng.csv \
  --data-dir ../data/bc5cdr-disease/processed_test \
  --dataset-name bc5cdr-disease-mesh \
  --mesh-separator "||"
```

### Example 2: Analyze with full UMLS Synonyms-Mesh Dictionary

```bash
python analyze_dictionary_coverage_flexible.py \
  --dict-path ../../utils/UMLS/full_mesh_eng.csv\
  --data-dir ../data/bc5cdr-chemical/processed_test \
  --dataset-name bc5cdr-chemical-mesh \
  --mesh-separator "||"  \
  --output-dir full_new_mesh_reports
```

```bash
python analyze_dictionary_coverage_flexible.py \
  --dict-path ../../utils/UMLS/full_mesh_eng.csv \
  --data-dir ../data/bc5cdr-disease/processed_test \
  --dataset-name bc5cdr-disease-mesh \
  --mesh-separator "||"  \
  --output-dir full_new_mesh_reports
```

### Example 3: Analyze with old UMLS Synonyms-Mesh Dictionary

```bash
python analyze_dictionary_coverage_flexible.py \
  --dict-path ../../utils/UMLS/synonyms-mesh.csv\
  --data-dir ../data/bc5cdr-chemical/processed_test \
  --dataset-name bc5cdr-chemical-mesh \
  --mesh-separator "||"  \
  --output-dir old_mesh_reports
```

```bash
python analyze_dictionary_coverage_flexible.py \
  --dict-path ../../utils/UMLS/synonyms-mesh.csv \
  --data-dir ../data/bc5cdr-disease/processed_test \
  --dataset-name bc5cdr-disease-mesh \
  --mesh-separator "||" \
  --output-dir old_mesh_reports
```

### Example 2: Analyze BC5CDR Dictionary (TXT format)

```bash
python analyze_dictionary_coverage_flexible.py \
  --dict-path ../data/bc5cdr-chemical/test_dictionary.txt \
  --data-dir ../data/bc5cdr-chemical/processed_test \
  --dataset-name bc5cdr-chemical-original
```

### Example 3: Custom CSV Dictionary

```bash
python analyze_dictionary_coverage_flexible.py \
  --dict-path my_custom_dict.csv \
  --data-dir ../data/bc5cdr-disease/processed_test \
  --dataset-name custom-disease-analysis \
  --output-dir custom_reports \
  --mesh-separator "||"
```

## Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--dict-path` | ✅ Yes | - | Path to dictionary file (`.txt` or `.csv`) |
| `--data-dir` | ✅ Yes | - | Path to processed test data directory |
| `--dataset-name` | ✅ Yes | - | Name for this analysis (used in output files) |
| `--output-dir` | No | `reports` | Directory to save output files |
| `--mesh-separator` | No | `\|\|` | Separator for MeSH CSV synonyms |

## Dictionary Formats

### BC5CDR Format (`.txt`)

**Format:** One CUI-term pair per line, separated by tab or space

```
D000001 atropine
D000001 atropine sulfate
D000002 acetaminophen
D000002 paracetamol
```

**Characteristics:**
- Simple text file
- One synonym per line
- Tab or space separator between CUI and term

### MeSH CSV Format (`.csv`)

**Format:** CSV with SourceID and Terms columns, synonyms separated by `||`

```csv
SourceID,Terms
D000001,atropine||atropine sulfate||Atropine Sulfate Anhydrous
D000002,acetaminophen||paracetamol||Tylenol||APAP
```

**Characteristics:**
- CSV file with header row
- Multiple synonyms in single cell
- Synonyms separated by `||` (customizable via `--mesh-separator`)

## Output

### Console Output

The analysis prints detailed statistics to the console:

```
================================================================================
DICTIONARY vs DATASET COVERAGE ANALYSIS: bc5cdr-chemical-mesh
================================================================================

📚 LOADING DICTIONARY...
Dictionary Total Entries: 62,905
Dictionary Unique CUIs: 10,603
Dictionary Unique Names: 62,905
Average Names per CUI: 5.93

📊 LOADING TEST DATASET...
Test Dataset Total Mentions: 5,015
Test Dataset Unique CUIs: 614

🎯 CUI COVERAGE ANALYSIS
Test CUIs in Dictionary: 507 / 614 (82.57%)
Test CUIs MISSING: 107 (17.43%)

💡 THEORETICAL MAXIMUM ACCURACY: 89.93%
```

### JSON Report

Each analysis generates a JSON file in the `reports/` directory:

```json
{
  "dataset_name": "bc5cdr-chemical-mesh",
  "dictionary_path": "mesh_chemicals_eng_fixed.csv",
  "dictionary": {
    "total_entries": 62905,
    "unique_cuis": 10603,
    "avg_names_per_cui": 5.93
  },
  "coverage": {
    "cui_coverage_pct": 82.57,
    "mention_coverage_pct": 0.00,
    "theoretical_max_accuracy_pct": 89.93
  },
  "missing": {
    "missing_cuis": 107,
    "impossible_queries": 505
  }
}
```

## Understanding the Results

### Key Metrics

**CUI Coverage**
- Percentage of test CUIs found in the dictionary
- Higher is better (>90% is excellent)
- Shows if the dictionary has the right concepts

**Mention Coverage (Exact Match)**
- Percentage of test mentions that exactly match dictionary terms
- Often 0% for cross-domain dictionaries (different terminology)
- Low values indicate need for semantic similarity matching

**Theoretical Maximum Accuracy**
- Best possible accuracy assuming perfect ranking
- Upper bound on performance with this dictionary
- Accounts for impossible queries (missing CUIs)

**Impossible Queries**
- Queries where NO CUI exists in the dictionary
- These cannot be answered correctly regardless of model
- Important for setting realistic accuracy expectations

### Interpreting Coverage

| CUI Coverage | Assessment | Recommendation |
|--------------|------------|----------------|
| >90% | ✅ Excellent | Dictionary is comprehensive |
| 80-90% | ⚠️ Good | Consider augmenting missing concepts |
| 70-80% | ⚠️ Fair | Significant gaps, augmentation needed |
| <70% | ❌ Poor | Dictionary mismatch, find alternatives |

## Common Scenarios

### Scenario 1: High CUI Coverage, Low Mention Coverage

**What it means:** Dictionary has the right concepts but uses different terminology

**Example:**
- CUI Coverage: 94%
- Mention Coverage: 0%

**Action:** Use semantic similarity models (e.g., SapBERT, BioLinkBERT) instead of exact matching

### Scenario 2: Missing Specialized Terms

**What it means:** Test set contains specialized terms not in standard dictionaries

**Example:** Research compounds, rare diseases, psychiatric disorders

**Action:** 
- Augment with domain-specific databases (ChEMBL, OMIM, DSM)
- Add commonly used lay terms
- Use UMLS broader vocabularies

### Scenario 3: Composite Annotations

**What it means:** Test annotations link to multiple concepts simultaneously

**Example:** `D003072D019964` (two disease CUIs concatenated)

**Action:** 
- Ensure dictionary has ALL component CUIs
- Consider partial credit in evaluation
- Review composite annotation handling in model

## Troubleshooting

### Error: "Could not detect ID/Terms columns"

**Cause:** CSV file doesn't have expected column names

**Solution:** Ensure CSV has columns named one of:
- ID columns: `SourceID`, `CUI`, `ID`, `Code`
- Terms columns: `Terms`, `Synonyms`, `Names`, `Aliases`

### Error: "Division by zero"

**Cause:** Dictionary loaded with 0 entries

**Solution:** 
- Check dictionary file format
- Verify file is not empty
- Ensure separator is correct (tab vs space for BC5CDR)

### Warning: "Average Names per CUI: 1.00"

**Cause:** MeSH CSV synonyms not being split (separator issue)

**Solution:** 
- Check if `||` separator exists in your CSV
- Try different separator with `--mesh-separator "SEPARATOR"`
- Verify CSV format is correct

### No Output Files Generated

**Cause:** `reports/` directory creation failed or path issues

**Solution:**
- Check write permissions in current directory
- Use `--output-dir` to specify alternative location
- Create `reports/` directory manually

## Advanced Usage

### Batch Analysis

Create a shell script for multiple dictionaries:

```bash
#!/bin/bash
# analyze_all.sh

DICTS=(
    "mesh_chemicals_eng.csv"
    "mesh_diseases_eng.csv"
    "snomed_chemicals.txt"
)

for dict in "${DICTS[@]}"; do
    python analyze_dictionary_coverage_flexible.py \
        --dict-path "$dict" \
        --data-dir ../data/bc5cdr-chemical/processed_test \
        --dataset-name "bc5cdr-chem-${dict%.*}"
done
```

### Custom Separators

For CSV files with different separators:

```bash
# Comma-separated
--mesh-separator ","

# Tab-separated
--mesh-separator $'\t'

# Pipe-separated
--mesh-separator "|"
```

### Programmatic Usage

```python
from analyze_dictionary_coverage_flexible import analyze_coverage

results = analyze_coverage(
    data_dir="../data/bc5cdr-chemical/processed_test",
    dict_path="my_dictionary.csv",
    dataset_name="my-analysis",
    output_dir="my_reports",
    mesh_separator="||"
)

print(f"CUI Coverage: {results['coverage']['cui_coverage_pct']:.2f}%")
print(f"Max Accuracy: {results['coverage']['theoretical_max_accuracy_pct']:.2f}%")
```

## File Locations Reference

### Standard Setup

```
PycharmProjects/sapbert/
├── utils/UMLS/
│   ├── mesh_chemicals_eng_fixed.csv
│   ├── mesh_chemicals_eng_bc5cdr_format.txt
│   ├── mesh_diseases_eng_fixed.csv
│   └── mesh_diseases_eng_bc5cdr_format.txt
│
├── evaluation/
│   ├── coverage_reports/
│   │   ├── README.md
│   │   ├── analyze_dictionary_coverage_flexible.py
│   │   ├── run_mesh_coverage_analysis.py
│   │   └── reports/
│   │       ├── bc5cdr-chemical-mesh_coverage_analysis.json
│   │       └── bc5cdr-disease-mesh_coverage_analysis.json
│   │
│   └── data/
│       ├── bc5cdr-chemical/
│       │   ├── processed_test/
│       │   └── test_dictionary.txt
│       └── bc5cdr-disease/
│           ├── processed_test/
│           └── test_dictionary.txt
```

## Related Tools

- **convert_mesh_to_bc5cdr_format.py**: Convert MeSH CSV to BC5CDR format
- **eval_sapbert_bc5cdr.py**: Run full BC5CDR evaluation with model

## Citation

If you use this tool in your research, please cite:

```bibtex
@article{liu2021sapbert,
  title={Self-Alignment Pretraining for Biomedical Entity Representations},
  author={Liu, Fangyu and Shareghi, Ehsan and Meng, Zaiqiao and Basaldella, Marco and Collier, Nigel},
  journal={arXiv preprint arXiv:2010.11784},
  year={2021}
}
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the code comments in `analyze_dictionary_coverage_flexible.py`
3. Open an issue in the repository

## License

This tool is part of the SapBERT project. See the main repository for license information.

---

**Last Updated:** January 4, 2026
**Version:** 1.0

