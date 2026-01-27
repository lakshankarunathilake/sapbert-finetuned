#!/bin/bash
# BC5CDR Retrieval Analysis Script
# Analyzes FAISS retrieval performance to understand reranking potential

python bc5cdr_analyze_retrieval.py \
  --base_model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
  --retriever_adapter_path /Users/lakshankarunathilake/Documents/Adapters/UMLS/sapbert-mesh-adapter \
  --faiss_index_path /Users/lakshankarunathilake/PycharmProjects/sapbert/utils/NEL/indexes/sapbert_bcd5cdr_decease_adapter_index/sapbert_bcd5cdr_decease_adapter_index \
  --pubtator_path /Users/lakshankarunathilake/PycharmProjects/sapbert/train/finetune/bc5cdr/CDR_Data/CDR.Corpus.v010516 \
  --category Disease \
  --splits train,validation,test \
  --k 10 \
  --output_csv retrieval_analysis_results.csv \
  --seed 13

