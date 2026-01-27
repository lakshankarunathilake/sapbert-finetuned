#!/bin/bash
# BC5CDR Reranker Training Script with Score Fusion
# Tests different fusion alpha values to find optimal balance between retrieval and reranker

# Base configuration
BASE_MODEL="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
PUBTATOR_PATH="/Users/lakshankarunathilake/PycharmProjects/sapbert/train/finetune/bc5cdr/CDR_Data/CDR.Corpus.v010516"
RETRIEVER_ADAPTER_PATH="/Users/lakshankarunathilake/Documents/Adapters/UMLS/sapbert-mesh-adapter"
FAISS_INDEX_PATH="/Users/lakshankarunathilake/PycharmProjects/sapbert/utils/NEL/indexes/sapbert_bcd5cdr_adapter_index/sapbert_bcd5cdr_adapter_index"
RERANK_ADAPTER_NAME="link_rerank"
CATEGORY="Disease"
EPOCHS=10
BATCH_SIZE=2
LR=5e-5
QUERY_MODE="context"
CONTEXT_WINDOW_CHARS=250

# Array of fusion alpha values to test
# Alpha represents weight for retrieval score:
#   1.0 = pure retrieval (no reranking)
#   0.9 = 90% retrieval, 10% reranker (very conservative)
#   0.8 = 80% retrieval, 20% reranker (recommended default)
#   0.7 = 70% retrieval, 30% reranker (balanced)
#   0.5 = 50% retrieval, 50% reranker (equal weight)
#   0.0 = pure reranker (original behavior, may hurt Acc@1)
FUSION_ALPHAS=(0.5 0.7 0.8 0.9 0.95)

echo "=========================================="
echo "BC5CDR Reranker Training - Fusion Alpha Grid Search"
echo "=========================================="
echo "Category: $CATEGORY"
echo "Epochs: $EPOCHS"
echo "Context window: $CONTEXT_WINDOW_CHARS chars"
echo "Fusion alphas: ${FUSION_ALPHAS[@]}"
echo ""
echo "Alpha interpretation:"
echo "  1.0 = Pure retrieval (no reranking)"
echo "  0.8 = 80% retrieval + 20% reranker"
echo "  0.5 = Equal weight"
echo "  0.0 = Pure reranker"
echo "=========================================="
echo ""

# Also test pure reranker (no fusion) as baseline
echo ""
echo "=========================================="
echo "Training BASELINE: Pure Reranker (no fusion)"
echo "=========================================="

OUTPUT_DIR="./out/rerank_fusion_baseline_alpha0.0"

python bc5cdr_rerank_train_fusion.py \
  --base_model "$BASE_MODEL" \
  --pubtator_path "$PUBTATOR_PATH" \
  --retriever_adapter_path "$RETRIEVER_ADAPTER_PATH" \
  --faiss_index_path "$FAISS_INDEX_PATH" \
  --rerank_adapter_name "$RERANK_ADAPTER_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --category "$CATEGORY" \
  --k 10 \
  --epochs "$EPOCHS" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --query_mode "$QUERY_MODE" \
  --context_window_chars "$CONTEXT_WINDOW_CHARS"

if [ $? -eq 0 ]; then
    echo "✅ Successfully completed baseline training (no fusion)"
else
    echo "❌ Error during baseline training"
    exit 1
fi

# Loop through each fusion alpha value
for ALPHA in "${FUSION_ALPHAS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Training with fusion_alpha=${ALPHA}"
    echo "=========================================="

    OUTPUT_DIR="./out/rerank_fusion_alpha${ALPHA}"

    python bc5cdr_rerank_train_fusion.py \
      --base_model "$BASE_MODEL" \
      --pubtator_path "$PUBTATOR_PATH" \
      --retriever_adapter_path "$RETRIEVER_ADAPTER_PATH" \
      --faiss_index_path "$FAISS_INDEX_PATH" \
      --rerank_adapter_name "$RERANK_ADAPTER_NAME" \
      --output_dir "$OUTPUT_DIR" \
      --category "$CATEGORY" \
      --k 10 \
      --epochs "$EPOCHS" \
      --per_device_train_batch_size "$BATCH_SIZE" \
      --lr "$LR" \
      --query_mode "$QUERY_MODE" \
      --context_window_chars "$CONTEXT_WINDOW_CHARS" \
      --use_fusion \
      --fusion_alpha "$ALPHA"

    if [ $? -eq 0 ]; then
        echo "✅ Successfully completed training for fusion_alpha=${ALPHA}"
    else
        echo "❌ Error during training for fusion_alpha=${ALPHA}"
        exit 1
    fi

    echo ""
done

echo ""
echo "=========================================="
echo "All training runs completed!"
echo "=========================================="
echo "Results saved in:"
echo "  - ./out/rerank_fusion_baseline_alpha0.0/ (no fusion)"
for ALPHA in "${FUSION_ALPHAS[@]}"; do
    echo "  - ./out/rerank_fusion_alpha${ALPHA}/"
done

echo ""
echo "=========================================="
echo "Next Steps: Compare Results"
echo "=========================================="
echo ""
echo "To compare all results, check these CSV files:"
echo "  ./out/rerank_fusion_baseline_alpha0.0/final_evaluation_results.csv"
for ALPHA in "${FUSION_ALPHAS[@]}"; do
    echo "  ./out/rerank_fusion_alpha${ALPHA}/final_evaluation_results.csv"
done
echo ""
echo "Or run the comparison script:"
echo "  python compare_fusion_results.py"