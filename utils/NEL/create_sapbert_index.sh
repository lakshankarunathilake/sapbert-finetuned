#!/bin/bash

# SAPBERT Index Creation Script
# Usage: ./create_sapbert_index.sh [csv_path] [output_dir] [index_name]
# Defaults are set to match your usual parameters.

# Default parameters
CSV_PATH="/Users/lakshankarunathilake/PycharmProjects/sapbert/evaluation/data/bc5cdr-disease/test_dictionary_converted.csv"
MODEL_NAME="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
ADAPTER_PATH="/Users/lakshankarunathilake/Documents/Adapters/UMLS/sapbert-mesh-adapter"
OUTPUT_DIR="/Users/lakshankarunathilake/PycharmProjects/sapbert/utils/NEL/indexes/sapbert_bcd5cdr_decease_adapter_index"
INDEX_NAME="sapbert_bcd5cdr_decease_adapter_index"
MAX_LENGTH=25
BATCH_SIZE=16
VERBOSE="--verbose"

# Allow overrides from command line
if [ -n "$1" ]; then
  CSV_PATH="$1"
fi
if [ -n "$2" ]; then
  OUTPUT_DIR="$2"
fi
if [ -n "$3" ]; then
  INDEX_NAME="$3"
fi

# Display configuration
echo "=========================================="
echo "SAPBERT Index Creation"
echo "=========================================="
echo "CSV Path: $CSV_PATH"
echo "Model Name: $MODEL_NAME"
echo "Adapter Path: $ADAPTER_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "Index Name: $INDEX_NAME"
echo "Max Length: $MAX_LENGTH"
echo "Batch Size: $BATCH_SIZE"
echo "Verbose: $VERBOSE"
echo "=========================================="
echo ""

# Run the python script
python create_sapbert_index.py \
  --csv_path "$CSV_PATH" \
  --model_name "$MODEL_NAME" \
  --adapter_path "$ADAPTER_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --index_name "$INDEX_NAME" \
  --max_length "$MAX_LENGTH" \
  --batch_size "$BATCH_SIZE" \
  $VERBOSE

