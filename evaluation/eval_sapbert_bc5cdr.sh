#!/bin/bash

# SAPBERT BC5CDR Evaluation Script
# Simple wrapper for eval_sapbert_bc5cdr.py
# Usage: ./eval_sapbert_bc5cdr.sh [chemical|disease]
#   Default: chemical

# Determine dataset type (default: chemical)
DATASET_TYPE="${1:-chemical}"

# Validate dataset type
if [[ "$DATASET_TYPE" != "chemical" && "$DATASET_TYPE" != "disease" ]]; then
    echo "Error: Invalid dataset type '$DATASET_TYPE'"
    echo "Usage: $0 [chemical|disease]"
    exit 1
fi

# Parameters
INDEX_PATH="/Users/lakshankarunathilake/PycharmProjects/sapbert/utils/NEL/indexes/mesh_adapter"
DATA_DIR="./data/bc5cdr-${DATASET_TYPE}"
ADAPTER_PATH="/Users/lakshankarunathilake/Documents/Adapters/UMLS/sapbert-mesh-adapter"

# Display configuration
echo "=========================================="
echo "BC5CDR Evaluation - ${DATASET_TYPE^^}"
echo "=========================================="
echo "Index Path: $INDEX_PATH"
echo "Data Dir: $DATA_DIR"
echo "Adapter Path: $ADAPTER_PATH"
echo ""

# Run evaluation
python eval_sapbert_bc5cdr.py \
    --index_path "$INDEX_PATH" \
    --data_dir "$DATA_DIR" \
    --adapter_path "$ADAPTER_PATH"
