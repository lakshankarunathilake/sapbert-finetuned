#!/bin/bash

# SAPBERT BC5CDR Evaluation Script
# Simple wrapper for eval_sapbert_bc5cdr.py
# Usage: ./eval_sapbert_bc5cdr.sh [chemical|disease] [topk]
#   Default: chemical, topk=10

# Determine dataset type (default: chemical)
DATASET_TYPE="${1:-chemical}"

# Determine topk (default: 10)
TOPK="${2:-10}"

# Validate dataset type
if [[ "$DATASET_TYPE" != "chemical" && "$DATASET_TYPE" != "disease" ]]; then
    echo "Error: Invalid dataset type '$DATASET_TYPE'"
    echo "Usage: $0 [chemical|disease] [topk]"
    echo "  dataset_type: chemical or disease (default: chemical)"
    echo "  topk: number of top candidates to retrieve (default: 10)"
    exit 1
fi

# Validate topk is a positive integer
if ! [[ "$TOPK" =~ ^[0-9]+$ ]] || [ "$TOPK" -lt 1 ]; then
    echo "Error: topk must be a positive integer (got: $TOPK)"
    echo "Usage: $0 [chemical|disease] [topk]"
    exit 1
fi

# Parameters
INDEX_PATH="/Users/lakshankarunathilake/PycharmProjects/sapbert/utils/NEL/indexes/sapbert_bcd5cdr_decease_adapter_index/sapbert_bcd5cdr_decease_adapter_index"
DATA_DIR="./data/bc5cdr-${DATASET_TYPE}"
ADAPTER_PATH="/Users/lakshankarunathilake/Documents/Adapters/UMLS/sapbert-mesh-adapter"

# Display configuration
echo "=========================================="
echo "BC5CDR Evaluation - ${DATASET_TYPE^^}"
echo "=========================================="
echo "Index Path: $INDEX_PATH"
echo "Data Dir: $DATA_DIR"
echo "Adapter Path: $ADAPTER_PATH"
echo "Top-K Candidates: $TOPK"
echo ""

# Run evaluation
python eval_sapbert_bc5cdr.py \
    --index_path "$INDEX_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "./evaluation_results/bc5cdr_${DATASET_TYPE}" \
    --adapter_path "$ADAPTER_PATH" \
    --topk "$TOPK"

