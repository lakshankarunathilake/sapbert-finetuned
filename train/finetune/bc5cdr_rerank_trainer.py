#!/usr/bin/env python

import argparse
import os
import pickle
import json
import glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

import torch
from torch import nn
import numpy as np
from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer, TrainingArguments
from adapters import AutoAdapterModel
from adapters.trainer import AdapterTrainer
from tqdm import tqdm


import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # → sapbert/
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append("/Users/lakshankarunathilake/PycharmProjects/sapbert/utils/NEL/create_sapbert_index.py")
from utils.NEL.search_sapbert_index import SAPBERTIndexSearcher

try:
    import faiss

    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ------------------------
# Args
# ------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--faiss_index_path", required=True, help="Path to FAISS index (without extension)")

    # Dataset options
    p.add_argument("--pubtator_path", required=True, help="Path to directory with PubTator format files (CDR_*.txt)")

    # Retriever adapter override (optional)
    p.add_argument("--retriever_adapter_name", default=None, help="Override adapter name from Hub for retriever")
    p.add_argument("--retriever_adapter_path", default=None, help="Override adapter local path for retriever")

    p.add_argument("--rerank_adapter_name", default="link_rerank", help="Name for reranker adapter stack")
    p.add_argument("--rerank_adapter_load", default=None, help="Load a pre-trained reranker adapter from a dir")

    p.add_argument("--output_dir", default="./out/reranker")

    p.add_argument("--k", type=int, default=50)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--query_mode", choices=["mention", "context"], default="context")
    p.add_argument("--context_window", type=int, default=64)

    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--per_device_train_batch_size", type=int, default=64)
    p.add_argument("--per_device_eval_batch_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=32, help="Batch size for reranking during evaluation")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=13)

    p.add_argument("--evaluate_only", action="store_true")
    p.add_argument("--train_split", default="validation", choices=["train", "validation"],
                   help="Which split to use for training")
    return p.parse_args()


# ------------------------
# PubTator Format Parser
# ------------------------

def parse_pubtator_file(filepath: str) -> List[Dict]:
    """
    Parse a single PubTator format file

    PubTator format:
    PMID|t|TITLE
    PMID|a|ABSTRACT
    PMID\tSTART\tEND\tMENTION\tTYPE\tMESH_ID

    Returns list of documents with entities
    """
    documents = []
    current_doc = None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                # Empty line marks end of document
                if current_doc and current_doc['entities']:
                    documents.append(current_doc)
                current_doc = None
                continue

            if '|t|' in line or '|a|' in line:
                # Title or abstract line
                parts = line.split('|')
                pmid = parts[0]
                text_type = parts[1]  # 't' for title, 'a' for abstract
                text = '|'.join(parts[2:]) if len(parts) > 2 else ''

                if current_doc is None or current_doc['id'] != pmid:
                    # Start new document
                    if current_doc and current_doc['entities']:
                        documents.append(current_doc)
                    current_doc = {
                        'id': pmid,
                        'passages': [],
                        'entities': [],
                        'full_text': ''
                    }

                current_doc['passages'].append({
                    'type': 'title' if text_type == 't' else 'abstract',
                    'text': text,
                    'offset': len(current_doc['full_text'])
                })
                if current_doc['full_text']:
                    current_doc['full_text'] += '\n'
                current_doc['full_text'] += text

            elif '\t' in line:
                # Entity annotation line
                parts = line.split('\t')
                if len(parts) >= 6:
                    pmid = parts[0]
                    start = int(parts[1])
                    end = int(parts[2])
                    mention = parts[3]
                    entity_type = parts[4]
                    mesh_ids = parts[5].split('|') if len(parts) > 5 else []

                    if current_doc and current_doc['id'] == pmid:
                        for mesh_id in mesh_ids:
                            if mesh_id and mesh_id != '-1':
                                current_doc['entities'].append({
                                    'text': mention,
                                    'start': start,
                                    'end': end,
                                    'type': entity_type,
                                    'mesh': mesh_id,
                                    'normalized': [{'db_name': 'MESH', 'db_id': mesh_id}]
                                })

        # Add last document
        if current_doc and current_doc['entities']:
            documents.append(current_doc)

    return documents


def load_bc5cdr_from_pubtator(pubtator_dir: str) -> Dict:
    """
    Load BC5CDR from PubTator format files

    Expected files in directory:
    - CDR_TrainingSet.PubTator.txt (or CDR_DevelopmentSet.PubTator.txt)
    - CDR_TestSet.PubTator.txt
    """
    print(f"📂 Loading BC5CDR from PubTator directory: {pubtator_dir}")

    splits = {}

    # Common file naming patterns
    file_patterns = {
        'train': ['*Training*.txt', '*train*.txt', '*Train*.txt'],
        'validation': ['*Development*.txt', '*dev*.txt', '*Dev*.txt', '*validation*.txt'],
        'test': ['*Test*.txt', '*test*.txt']
    }

    for split, patterns in file_patterns.items():
        found = False
        for pattern in patterns:
            files = glob.glob(os.path.join(pubtator_dir, pattern))
            if files:
                filepath = files[0]
                print(f"  📄 Loading {split} from: {os.path.basename(filepath)}")
                docs = parse_pubtator_file(filepath)

                # Convert to dataset format
                splits[split] = {
                    'id': [d['id'] for d in docs],
                    'passages': [d['passages'] for d in docs],
                    'entities': [d['entities'] for d in docs],
                    'full_text': [d['full_text'] for d in docs]
                }
                print(f"    ✅ Loaded {len(docs)} documents with {sum(len(d['entities']) for d in docs)} entities")
                found = True
                break

        if not found and split != 'validation':
            print(f"  ⚠️  No {split} file found")

    # If no validation set, try using dev set
    if 'validation' not in splits:
        print(f"  ℹ️  No validation set found, will use train split for training")

    if not splits:
        raise RuntimeError(
            f"No PubTator files found in {pubtator_dir}\nExpected files like: CDR_TrainingSet.PubTator.txt, CDR_TestSet.PubTator.txt")

    return splits


# ------------------------
# Data helpers
# ------------------------


def load_bc5cdr(pubtator_path: str):
    """Load BC5CDR dataset from PubTator format files"""
    print("📥 Loading BC5CDR dataset from PubTator format...")
    print(f"  📂 Directory: {pubtator_path}")

    # Load from PubTator format
    splits_dict = load_bc5cdr_from_pubtator(pubtator_path)

    # Convert to simple dict-based dataset (no HuggingFace dependency)
    ds = {}
    for split_name, split_data in splits_dict.items():
        # Just keep as dict of lists
        ds[split_name] = split_data
        print(f"  ✅ {split_name}: {len(split_data['id'])} documents")

    # Ensure required splits
    if 'validation' not in ds and 'train' in ds:
        print("  ⚠️  No validation set found, creating from train split (10%)")
        # Simple split without HuggingFace
        train_data = ds['train']
        num_train = len(train_data['id'])
        split_idx = int(num_train * 0.9)

        # Split train into train and validation
        val_data = {k: v[split_idx:] for k, v in train_data.items()}
        train_data_new = {k: v[:split_idx] for k, v in train_data.items()}

        ds['train'] = train_data_new
        ds['validation'] = val_data
        print(f"    ✅ New train: {len(train_data_new['id'])} documents")
        print(f"    ✅ New validation: {len(val_data['id'])} documents")

    # Check required splits
    missing = []
    for split in ['train', 'validation', 'test']:
        if split not in ds:
            missing.append(split)

    if missing:
        raise RuntimeError(f"Missing required splits: {missing}. Available: {list(ds.keys())}")

    return ds


def collect_mentions(ds_split, split_name: str = "split") -> List[dict]:
    """Collect mentions from dataset split with progress (dict-of-lists input)."""
    mentions = []
    print(f"📋 Collecting mentions from {split_name} split...")

    # Expect keys: id, passages, entities, full_text (each a list of len N)
    n = len(ds_split["id"])
    for idx in tqdm(range(n), desc=f"Processing {split_name}", leave=False):
        # materialize an example dict
        ex = {k: ds_split[k][idx] for k in ds_split.keys()}

        passages = ex.get("passages", [])
        entities = ex.get("entities", [])
        full_text = ex.get("full_text", None)

        # Build document text
        if full_text:
            doc_text = full_text
        else:
            parts = []
            for p in passages:
                t = p.get("text", "") if isinstance(p, dict) else p["text"]
                if isinstance(t, list):
                    parts.extend(t)
                elif isinstance(t, str):
                    parts.append(t)
            doc_text = "\n".join(parts) if parts else None

        # Entities -> mentions
        for ent in entities:
            mesh = None
            if isinstance(ent, dict) and 'mesh' in ent:
                mesh = ent['mesh']

            if not mesh and 'normalized' in ent:
                norm = ent.get("normalized", [])
                if norm:
                    for nrm in norm:
                        if isinstance(nrm, dict) and str(nrm.get("db_name", "")).lower() == "mesh":
                            mesh = nrm.get("db_id"); break
                    if mesh is None and norm and isinstance(norm[0], dict):
                        mesh = norm[0].get("db_id")

            if mesh is None:
                continue

            text = ent.get("text", "")
            if isinstance(text, list):
                text = " ".join(text)
            if not text:
                continue

            mentions.append({"text": text, "mesh": mesh, "doc_text": doc_text})

    return mentions



def build_context_query(mention: dict, tokenizer, window_tokens: int) -> str:
    """Build context-aware query from mention"""
    text = mention.get("text") or ""
    doc = mention.get("doc_text")
    if not doc:
        return text
    try:
        start = doc.lower().index(text.lower())
        end = start + len(text)
    except ValueError:
        return text
    left = doc[:start];
    right = doc[end:]
    lt = tokenizer.tokenize(left)[-window_tokens:]
    rt = tokenizer.tokenize(right)[:window_tokens]
    lstr = tokenizer.convert_tokens_to_string(lt).strip()
    rstr = tokenizer.convert_tokens_to_string(rt).strip()
    return f"{lstr} [MENTION] {text} [/MENTION] {rstr}".strip()


# ------------------------
# Listwise dataset for AdapterTrainer
# ------------------------

@dataclass
class RerankExample:
    input_ids: List[int]
    attention_mask: List[int]
    query_id: int
    label: int  # 1 for gold, 0 for negatives


def make_listwise_dataset(tokenizer: AutoTokenizer, pairs: List[Tuple[str, List[str]]], max_length: int) -> HFDataset:
    """Build listwise dataset from query-candidate pairs with progress"""
    features: List[Dict] = []

    print(f"🏗️  Building training dataset from {len(pairs):,} queries...")

    for qid, (q, cand_list) in enumerate(tqdm(pairs, desc="Tokenizing pairs")):
        for j, alias in enumerate(cand_list):
            enc = tokenizer(f"{q} [SEP] {alias}", truncation=True, padding=False, max_length=max_length)
            features.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "query_id": qid,
                "label": 1 if j == 0 else 0,
            })

    print(f"  ✅ Created {len(features):,} training examples")
    return HFDataset.from_list(features)


# ------------------------
# Custom AdapterTrainer implementing group (listwise) softmax per query_id
# ------------------------

class ListwiseAdapterTrainer(AdapterTrainer):
    """Custom trainer with listwise softmax loss"""

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        # accept both 'labels' and 'label'
        if "labels" in inputs:
            labels = inputs.pop("labels")
        elif "label" in inputs:
            labels = inputs.pop("label")
        else:
            raise KeyError(f"No label field in inputs. Got keys: {list(inputs.keys())}")

        if "query_id" not in inputs:
            raise KeyError(f"'query_id' missing from inputs. Got keys: {list(inputs.keys())}")

        qids = inputs.pop("query_id")

        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)

        # 1-D ensure
        if hasattr(qids, "ndim") and qids.ndim > 1:
            qids = qids.view(-1)
        if hasattr(labels, "ndim") and labels.ndim > 1:
            labels = labels.view(-1)

        loss_terms = []
        for qid in torch.unique(qids):
            mask = (qids == qid)
            group_logits = logits[mask]
            group_labels = labels[mask]

            log_probs = torch.log_softmax(group_logits, dim=0)
            pos_idx = (group_labels == 1).nonzero(as_tuple=True)[0]
            if pos_idx.numel() == 0:
                continue
            assert pos_idx.numel() == 1, f"Expected 1 positive per query, got {pos_idx.numel()}"
            loss_terms.append(-log_probs[pos_idx[0]])

        if loss_terms:
            loss = torch.stack(loss_terms).mean()
        else:
            # No positives in this batch → return a zero *tensor* tied to logits
            loss = logits.sum() * 0.0

        return (loss, outputs) if return_outputs else loss



# ------------------------
# Retrieval and evaluation helpers
# ------------------------

def extract_candidates_from_search_results(search_results: List[List[Dict]]) -> List[Tuple[List[str], List[str]]]:
    """Extract (aliases, entity_ids) from search results"""
    candidates = []
    for results in search_results:
        cand_aliases = []
        cand_ids = []
        for r in results:
            entity_id = r.get('entity_id')
            # Get alias from various possible keys
            alias = (r.get('primary_alias') or
                     r.get('processed_text') or
                     (r.get('aliases', [''])[0] if r.get('aliases') else '') or
                     (r.get('all_aliases', [''])[0] if r.get('all_aliases') else ''))
            if entity_id and alias:
                cand_aliases.append(alias)
                cand_ids.append(entity_id)
        candidates.append((cand_aliases, cand_ids))
    return candidates


def evaluate_reranker(rer_model, tokenizer, test_mentions, test_queries,
                      candidates_per_query: List[Tuple[List[str], List[str]]],
                      max_length, device, batch_size=32):
    """Comprehensive evaluation with batched reranking and progress bars"""
    print(f"📊 Evaluating reranker on {len(test_mentions):,} test mentions...")
    rer_model.eval()

    metrics = {
        'retrieval_recall': 0,
        'top1': 0,
        'top5': 0,
        'mrr': 0.0,
        'counted': 0
    }

    # Batched evaluation for speed
    num_batches = (len(test_mentions) + batch_size - 1) // batch_size
    pbar = tqdm(range(0, len(test_mentions), batch_size), desc="Reranking batches", total=num_batches)

    for batch_start in pbar:
        batch_end = min(batch_start + batch_size, len(test_mentions))
        batch_mentions = test_mentions[batch_start:batch_end]
        batch_queries = test_queries[batch_start:batch_end]
        batch_candidates = candidates_per_query[batch_start:batch_end]

        batch_pairs = []
        batch_metadata = []

        for i, m in enumerate(batch_mentions):
            gold = m["mesh"]
            cand_aliases, cand_cids = batch_candidates[i]

            if gold not in cand_cids:
                continue

            metrics['retrieval_recall'] += 1
            metrics['counted'] += 1
            query_text = batch_queries[i]

            pairs = [f"{query_text} [SEP] {a}" for a in cand_aliases]
            start_idx = len(batch_pairs)
            batch_pairs.extend(pairs)
            end_idx = len(batch_pairs)

            batch_metadata.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'gold': gold,
                'cand_cids': cand_cids
            })

        if not batch_pairs:
            continue

        inputs = tokenizer(batch_pairs, padding=True, truncation=True,
                           max_length=max_length, return_tensors="pt").to(device)

        with torch.no_grad():
            all_logits = rer_model(**inputs).logits.squeeze(-1)

        for meta in batch_metadata:
            query_logits = all_logits[meta['start_idx']:meta['end_idx']]
            ranked_indices = torch.argsort(query_logits, descending=True).cpu().tolist()
            ranked_cids = [meta['cand_cids'][idx] for idx in ranked_indices]

            if ranked_cids[0] == meta['gold']:
                metrics['top1'] += 1
            if meta['gold'] in ranked_cids[:5]:
                metrics['top5'] += 1

            try:
                rank = ranked_cids.index(meta['gold']) + 1
                metrics['mrr'] += 1.0 / rank
            except ValueError:
                pass

        pbar.set_postfix({
            'recall': f"{metrics['retrieval_recall']}/{batch_end}",
            'top1': f"{metrics['top1']}/{metrics['counted']}"
        })

    total_mentions = len(test_mentions)
    counted = max(1, metrics['counted'])

    print(f"  ✅ Evaluation complete")

    return {
        'retrieval_recall@k': metrics['retrieval_recall'] / total_mentions,
        'rerank_top1': metrics['top1'] / counted,
        'rerank_top5': metrics['top5'] / counted,
        'mrr': metrics['mrr'] / counted,
        'evaluated_on': f"{metrics['counted']}/{total_mentions}",
        'gold_in_topk': metrics['retrieval_recall']
    }


# ------------------------
# Main
# ------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Validate FAISS
    if not HAS_FAISS:
        raise RuntimeError("FAISS not available. Install with: pip install faiss-cpu (or faiss-gpu)")

    # Device handling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70)
    print("BC5CDR RERANKER TRAINER (using SAPBERTIndexSearcher)".center(70))
    print("=" * 70)
    print(f"🖥️  Device: {device}")
    print(f"🌱 Seed: {args.seed}")
    print()

    # ============================================================
    # STEP 1: Load BC5CDR dataset
    # ============================================================
    print("\n" + "=" * 70)
    print("[STEP 1/7] LOAD BC5CDR DATASET")
    print("=" * 70)
    ds = load_bc5cdr(pubtator_path=args.pubtator_path)
    print(f"  ✅ Train: {len(ds['train']):,} documents")
    print(f"  ✅ Validation: {len(ds['validation']):,} documents")
    print(f"  ✅ Test: {len(ds['test']):,} documents")

    # ============================================================
    # STEP 2: Load FAISS index with SAPBERTIndexSearcher
    # ============================================================
    print("\n" + "=" * 70)
    print("[STEP 2/7] INITIALIZE SAPBERT INDEX SEARCHER")
    print("=" * 70)

    # Initialize searcher with optional adapter override
    searcher = SAPBERTIndexSearcher(
        model_name=args.base_model,
        adapter_name=args.retriever_adapter_name,
        adapter_path=args.retriever_adapter_path
    )

    # Load index
    searcher.load_index(args.faiss_index_path)
    searcher.get_index_stats()


    # ============================================================
    # STEP 3: Prepare training data
    # ============================================================
    print("\n" + "=" * 70)
    print(f"[STEP 3/7] PREPARE {args.train_split.upper()} SPLIT FOR TRAINING")
    print("=" * 70)

    train_mentions = collect_mentions(ds[args.train_split], args.train_split)
    print(f"  ✅ Collected {len(train_mentions):,} mentions")

    # Get tokenizer from searcher for context building
    searcher._load_model()  # Ensure model/tokenizer loaded

    def to_query(m):
        if args.query_mode == "context":
            return build_context_query(m, searcher.tokenizer, args.context_window)
        return m["text"]

    print(f"🔄 Building queries (mode: {args.query_mode})...")
    train_queries = [to_query(m) for m in tqdm(train_mentions, desc="Building queries", leave=False)]

    # Use searcher's batch_search method
    print(f"🔍 Retrieving top-{args.k} candidates using SAPBERTIndexSearcher...")
    train_search_results = searcher.batch_search(train_queries, k=args.k)
    train_candidates = extract_candidates_from_search_results(train_search_results)

    # Build training pairs
    print(f"🔗 Building training pairs (gold + negatives)...")
    pairs: List[Tuple[str, List[str]]] = []
    kept = 0

    for i in tqdm(range(len(train_mentions)), desc="Creating pairs", leave=False):
        m = train_mentions[i]
        gold = m["mesh"]
        cand_aliases, cand_cids = train_candidates[i]
        if gold not in cand_cids:
            continue
        gold_alias = cand_aliases[cand_cids.index(gold)]
        negs = [a for a, c in zip(cand_aliases, cand_cids) if c != gold]
        pairs.append((train_queries[i], [gold_alias] + negs))
        kept += 1

    print(f"  ✅ Created {kept:,} training groups (gold within top-{args.k})")
    print(f"  ✅ Training recall@{args.k}: {kept / len(train_mentions) * 100:.2f}%")

    # ============================================================
    # STEP 4: Build reranker model and dataset
    # ============================================================
    print("\n" + "=" * 70)
    print("[STEP 4/7] BUILD RERANKER ADAPTER & DATASET")
    print("=" * 70)

    print(f"🔧 Loading base model: {args.base_model}")
    rer_model = AutoAdapterModel.from_pretrained(args.base_model)

    if args.rerank_adapter_load:
        print(f"📦 Loading pre-trained reranker from: {args.rerank_adapter_load}")
        rer_model.load_adapter(args.rerank_adapter_load, load_as=args.rerank_adapter_name, with_head=True)
    else:
        print(f"✨ Creating new reranker adapter: {args.rerank_adapter_name}")
        rer_model.add_adapter(args.rerank_adapter_name)
        if f"{args.rerank_adapter_name}_head" not in rer_model.config.prediction_heads:
            rer_model.add_classification_head(args.rerank_adapter_name, num_labels=1)

    rer_model.set_active_adapters([args.rerank_adapter_name])
    rer_model.train_adapter(args.rerank_adapter_name)
    rer_model.to(device)
    print(f"  ✅ Reranker loaded on {device}")

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    train_ds = make_listwise_dataset(tok, pairs, max_length=args.max_length)

    args_hf = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        weight_decay=args.weight_decay,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="no",
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
        use_cpu=True

    )

    # ============================================================
    # STEP 5: Train reranker
    # ============================================================
    print("\n" + "=" * 70)
    print("[STEP 5/7] TRAIN RERANKER ADAPTER")
    print("=" * 70)

    trainer = ListwiseAdapterTrainer(
        model=rer_model,
        args=args_hf,
        train_dataset=train_ds,
        tokenizer=tok,
    )

    if not args.evaluate_only and args.epochs > 0:
        print(f"🚀 Starting training ({args.epochs} epochs)...")
        print(f"   Batch size: {args.per_device_train_batch_size}")
        print(f"   Learning rate: {args.lr}")
        print(f"   Optimizer: AdamW (weight_decay={args.weight_decay})")
        print()
        trainer.train()
        print(f"\n💾 Saving model to: {args.output_dir}")
        trainer.save_model(args.output_dir)
        rer_model.save_adapter(args.output_dir, args.rerank_adapter_name)
        print(f"  ✅ Model saved successfully")
    else:
        print("  ⏭️  Skipping training (evaluate_only or epochs=0)")

    # ============================================================
    # STEP 6: Evaluate on test set
    # ============================================================
    print("\n" + "=" * 70)
    print("[STEP 6/7] EVALUATE ON TEST SET")
    print("=" * 70)

    test_mentions = collect_mentions(ds["test"], "test")
    test_queries = [to_query(m) for m in tqdm(test_mentions, desc="Building test queries", leave=False)]

    # Use searcher for test retrieval
    print(f"🔍 Retrieving top-{args.k} test candidates using SAPBERTIndexSearcher...")
    test_search_results = searcher.batch_search(test_queries, k=args.k)
    test_candidates = extract_candidates_from_search_results(test_search_results)

    # Batched evaluation
    results = evaluate_reranker(
        rer_model, tok, test_mentions, test_queries, test_candidates,
        args.max_length, device, batch_size=args.eval_batch_size
    )

    # ============================================================
    # Final Results
    # ============================================================
    print("\n" + "=" * 70)
    print("FINAL EVALUATION RESULTS".center(70))
    print("=" * 70)
    print(f"📊 Dataset:                 {results['evaluated_on']}")
    print(f"🔍 Retrieval Recall@{args.k:>2}:      {results['retrieval_recall@k'] * 100:>6.2f}%  (gold in top-K)")
    print("-" * 70)
    print(f"🥇 Reranker Top-1:          {results['rerank_top1'] * 100:>6.2f}%")
    print(f"🏅 Reranker Top-5:          {results['rerank_top5'] * 100:>6.2f}%")
    print(f"📈 Mean Reciprocal Rank:    {results['mrr']:>6.4f}")
    print("=" * 70)
    print(f"💾 Model saved in: {args.output_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()