#!/usr/bin/env python

import argparse
import os
import glob
from typing import Dict, List, Tuple
import logging
import random
import csv
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, TrainingArguments, TrainerCallback
from adapters import AutoAdapterModel
from adapters.trainer import AdapterTrainer
from tqdm import tqdm

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # → sapbert/
sys.path.insert(0, str(PROJECT_ROOT))
from utils.NEL.search_sapbert_index import SAPBERTIndexSearcher

# Import write_results_to_csv from bc5cdr_rerank_eval
from train.finetune.bc5cdr_rerank_eval import write_results_to_csv

try:
    import faiss

    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("⚠️  wandb not installed. Install with: pip install wandb")

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ------------------------
# Helper Functions
# ------------------------

def normalize_mesh_id(mesh_id: str) -> set:
    """
    Normalize MESH ID by splitting composites and filtering valid IDs

    Args:
        mesh_id: MESH ID, possibly composite like '184900|C566112'

    Returns:
        Set of valid MESH IDs (starting with C or D)
    """
    if not mesh_id:
        return set()

    ids = mesh_id.split('|')
    valid_ids = {id.strip() for id in ids if id.strip() and (id.strip().startswith('C') or id.strip().startswith('D'))}
    return valid_ids


def check_mesh_match(predicted_id: str, gold_id: str) -> bool:
    """
    Check if predicted MESH ID matches gold MESH ID
    Handles composite IDs by checking if any component matches
    """
    predicted_set = normalize_mesh_id(predicted_id)
    gold_set = normalize_mesh_id(gold_id)
    return len(predicted_set.intersection(gold_set)) > 0


# ------------------------
# Args
# ------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--faiss_index_path", required=True, help="Path to FAISS index (without extension)")
    p.add_argument("--pubtator_path", required=True, help="Path to directory with PubTator format files (CDR_*.txt)")

    p.add_argument("--retriever_adapter_name", default=None, help="Override adapter name from Hub for retriever")
    p.add_argument("--retriever_adapter_path", default=None, help="Override adapter local path for retriever")

    p.add_argument("--rerank_adapter_name", default="link_rerank", help="Name for reranker adapter stack")
    p.add_argument("--rerank_adapter_load", default=None, help="Load a pre-trained reranker adapter from a dir")

    p.add_argument("--output_dir", default="./out/reranker")

    p.add_argument("--k", type=int, default=10, help="(Ignored) always forced to 10 for this script")

    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--query_mode", choices=["mention", "context"], default="context")
    p.add_argument("--context_window_chars", type=int, default=256,
                   help="Number of characters to include on each side of mention for context (default: 256)")

    # NEW: Fusion parameters
    p.add_argument("--use_fusion", action="store_true",
                   help="Enable score fusion (blend retrieval + reranker scores)")
    p.add_argument("--fusion_alpha", type=float, default=0.8,
                   help="Fusion weight for retrieval score (0.8 = 80%% retrieval, 20%% reranker)")
    p.add_argument("--tune_alpha", action="store_true",
                   help="Auto-tune fusion_alpha on validation set (tests 0.5, 0.7, 0.8, 0.9, 0.95)")

    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--per_device_train_batch_size", type=int, default=64)
    p.add_argument("--per_device_eval_batch_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=32)

    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=13)

    p.add_argument("--evaluate_only", action="store_true")
    p.add_argument("--train_split", default="train", choices=["train", "validation"],
                   help="Which split to use for training (default: train)")
    p.add_argument("--category", type=str, choices=["Disease", "Chemical", "Both"], default="Both",
                   help="Select which category to use: Disease, Chemical, or Both (default: Both)")
    return p.parse_args()


# ------------------------
# PubTator Parser
# ------------------------

def parse_pubtator_file(filepath: str) -> List[Dict]:
    documents = []
    current_doc = None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_doc and current_doc['entities']:
                    documents.append(current_doc)
                current_doc = None
                continue

            if '|t|' in line or '|a|' in line:
                parts = line.split('|')
                pmid = parts[0]
                text_type = parts[1]
                text = '|'.join(parts[2:]) if len(parts) > 2 else ''

                if current_doc is None or current_doc['id'] != pmid:
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
                    current_doc['full_text'] += ' '
                current_doc['full_text'] += text

            elif '\t' in line:
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
                                })

        if current_doc and current_doc['entities']:
            documents.append(current_doc)

    return documents


def load_bc5cdr_from_pubtator(pubtator_dir: str) -> Dict:
    print(f"📂 Loading BC5CDR from PubTator directory: {pubtator_dir}")
    splits = {}
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

    if not splits:
        raise RuntimeError(f"No PubTator files found in {pubtator_dir}")

    return splits


def load_bc5cdr(pubtator_path: str) -> Dict:
    print("📥 Loading BC5CDR dataset from PubTator format...")
    print(f"  📂 Directory: {pubtator_path}")

    splits_dict = load_bc5cdr_from_pubtator(pubtator_path)
    ds = {}
    for split_name, split_data in splits_dict.items():
        ds[split_name] = split_data
        print(f"  ✅ {split_name}: {len(split_data['id'])} documents")

    missing = [s for s in ['train', 'validation', 'test'] if s not in ds]
    if missing:
        raise RuntimeError(f"Missing required splits: {missing}. Available: {list(ds.keys())}")

    return ds


# ------------------------
# Mention Collector
# ------------------------

def collect_mentions(ds_split, split_name: str = "split", category: str = "Both") -> List[dict]:
    mentions = []
    print(f"📋 Collecting mentions from {split_name} split...")

    n = len(ds_split["id"])
    for idx in tqdm(range(n), desc=f"Processing {split_name}", leave=False):
        ex = {k: ds_split[k][idx] for k in ds_split.keys()}
        entities = ex.get("entities", [])
        doc_text = ex.get("full_text", None)
        doc_id = ex.get("id", None)

        for ent in entities:
            mesh = ent.get("mesh")
            text = ent.get("text", "")
            ent_type = ent.get("type", "")
            if not mesh or not text:
                continue
            if category != "Both" and ent_type.lower() != category.lower():
                continue
            mentions.append({
                "text": text,
                "mesh": mesh,
                "doc_text": doc_text,
                "start": ent.get("start", None),
                "end": ent.get("end", None),
                "doc_id": doc_id
            })

    return mentions


# ------------------------
# Query Builder Functions
# ------------------------

def to_retrieval_query(m: dict) -> str:
    """Build query for FAISS retrieval (plain text, no markers)"""
    return m.get("text", "")


def to_rerank_query(m: dict, window_chars: int = 200) -> str:
    """Build query for reranking with context and markers"""
    return build_rerank_query_with_context(m, window_chars=window_chars)


def build_rerank_query_with_context(m: dict, window_chars: int = 200) -> str:
    """
    Build a context-aware query for reranking with [MENTION] markers
    """
    doc = m.get("doc_text") or ""
    start, end = m.get("start"), m.get("end")
    mention = m.get("text", "")

    if start is None or end is None or not doc:
        return f"[MENTION] {mention} [/MENTION]"

    if start < 0 or end > len(doc) or start >= end:
        return f"[MENTION] {mention} [/MENTION]"

    slice_ = doc[start:end]
    slice_normalized = " ".join(slice_.split())
    mention_normalized = " ".join(mention.split())

    if slice_ != mention:
        if slice_normalized != mention_normalized:
            search_start = max(0, start - 50)
            search_end = min(len(doc), end + 50)
            search_region = doc[search_start:search_end]

            idx = search_region.find(mention)
            if idx != -1:
                start = search_start + idx
                end = start + len(mention)
            else:
                logger.warning(f"Offset mismatch: expected '{slice_}' but got '{mention}' at [{start}:{end}]")
                return f"[MENTION] {mention} [/MENTION]"

    left = max(0, start - window_chars)
    right = min(len(doc), end + window_chars)

    left_ctx = doc[left:start]
    mid = doc[start:end]
    right_ctx = doc[end:right]

    left_ctx = " ".join(left_ctx.split())
    right_ctx = " ".join(right_ctx.split())
    mid = " ".join(mid.split())

    return f"{left_ctx} [MENTION] {mid} [/MENTION] {right_ctx}"


# ------------------------
# Dataset
# ------------------------

class SimpleDataset:
    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def make_listwise_dataset(tokenizer: AutoTokenizer,
                          pairs: List[Tuple[str, List[str]]],
                          max_length: int) -> SimpleDataset:
    features: List[Dict] = []
    print(f"🏗️  Building training dataset from {len(pairs):,} queries...")

    sep_token = tokenizer.sep_token if tokenizer.sep_token else "[SEP]"

    for qid, (q, cand_list) in enumerate(tqdm(pairs, desc="Tokenizing pairs")):
        for j, alias in enumerate(cand_list):
            enc = tokenizer(
                f"{q} {sep_token} {alias}",
                truncation=True,
                padding=False,
                max_length=max_length
            )
            features.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "query_id": qid,
                "label": 1 if j == 0 else 0,
            })

    print(f"  ✅ Created {len(features):,} training examples")
    return SimpleDataset(features)


# ------------------------
# Collator + Sampler
# ------------------------

class QueryGroupedCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, items: List[Dict]):
        input_ids = [it['input_ids'] for it in items]
        attention_mask = [it['attention_mask'] for it in items]
        query_ids = [it['query_id'] for it in items]
        labels = [it['label'] for it in items]

        max_len = max(len(ids) for ids in input_ids)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        input_ids_padded, attention_mask_padded = [], []
        for ids, mask in zip(input_ids, attention_mask):
            pad_len = max_len - len(ids)
            input_ids_padded.append(ids + [pad_id] * pad_len)
            attention_mask_padded.append(mask + [0] * pad_len)

        return {
            'input_ids': torch.tensor(input_ids_padded, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask_padded, dtype=torch.long),
            'query_id': torch.tensor(query_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }


class QueryGroupedSampler:
    """Yield batches that contain whole query groups"""

    def __init__(self, dataset: SimpleDataset, groups_per_batch: int = 1):
        groups = {}
        for idx, item in enumerate(dataset.data):
            qid = item["query_id"]
            groups.setdefault(qid, []).append(idx)
        self.group_list = list(groups.values())
        self.groups_per_batch = max(1, int(groups_per_batch))

    def __iter__(self):
        random.shuffle(self.group_list)
        batch = []
        count = 0
        for g in self.group_list:
            batch.extend(g)
            count += 1
            if count == self.groups_per_batch:
                yield batch
                batch = []
                count = 0
        if batch:
            yield batch

    def __len__(self):
        from math import ceil
        return ceil(len(self.group_list) / self.groups_per_batch)


# ------------------------
# Custom Trainer
# ------------------------

class ListwiseAdapterTrainer(AdapterTrainer):
    def __init__(self, *args, grouped_sampler=None, data_collator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._grouped_sampler = grouped_sampler
        self._data_collator = data_collator

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Training requires a train_dataset.")
        return DataLoader(
            self.train_dataset,
            batch_sampler=self._grouped_sampler,
            collate_fn=self._data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        labels = inputs.pop("labels", inputs.pop("label", None))
        qids = inputs.pop("query_id")
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)

        qids = qids.view(-1)
        labels = labels.view(-1)

        loss_terms = []
        for qid in torch.unique(qids):
            mask = (qids == qid)
            group_logits = logits[mask]
            group_labels = labels[mask]

            pos_idx = (group_labels == 1).nonzero(as_tuple=True)[0]
            if pos_idx.numel() != 1:
                continue

            log_probs = torch.log_softmax(group_logits, dim=0)
            loss_terms.append(-log_probs[pos_idx[0]])

        loss = torch.stack(loss_terms).mean() if loss_terms else logits.sum() * 0.0
        return (loss, outputs) if return_outputs else loss


# ------------------------
# NEW: Score Extraction from Search Results
# ------------------------

def extract_candidates_and_scores(search_results: List[List[Dict]]) -> Tuple[
    List[Tuple[List[str], List[str]]], List[List[float]]]:
    """
    Extract candidate aliases, IDs, AND retrieval scores from search results

    Returns:
        candidates: List of (alias_list, id_list) tuples
        scores: List of score lists (one per query)
    """
    candidates = []
    all_scores = []

    for results in search_results:
        cand_aliases, cand_ids, cand_scores = [], [], []
        for r in results:
            entity_id = r.get('entity_id')
            alias = (r.get('primary_alias') or
                     r.get('processed_text') or
                     (r.get('aliases', [''])[0] if r.get('aliases') else '') or
                     (r.get('all_aliases', [''])[0] if r.get('all_aliases') else ''))

            # Extract score - try multiple possible keys (including similarity_score from SAPBERTIndexSearcher)
            score = r.get('score', r.get('similarity', r.get('similarity_score', r.get('distance', None))))

            if entity_id and alias:
                cand_aliases.append(alias)
                cand_ids.append(entity_id)
                # If no score, use 0.0 as fallback (will disable fusion for this query)
                cand_scores.append(float(score) if score is not None else 0.0)

        candidates.append((cand_aliases, cand_ids))
        all_scores.append(cand_scores)

    return candidates, all_scores


# ------------------------
# NEW: Evaluation with Score Fusion
# ------------------------

def evaluate_reranker_with_fusion(rer_model, tokenizer, test_mentions, test_queries,
                                  candidates_per_query: List[Tuple[List[str], List[str]]],
                                  retrieval_scores: List[List[float]],
                                  max_length, device, batch_size=32,
                                  fusion_alpha=0.8, use_fusion=True):
    """
    Evaluate reranker with optional score fusion

    Args:
        fusion_alpha: Weight for retrieval score (0.8 = 80% retrieval, 20% reranker)
        use_fusion: If False, uses pure reranker scores (original behavior)
    """
    print(f"📊 Evaluating reranker on {len(test_mentions):,} test mentions...")
    if use_fusion:
        print(
            f"   🔀 Fusion enabled: α={fusion_alpha:.2f} (retrieval={fusion_alpha:.0%}, reranker={1 - fusion_alpha:.0%})")
    else:
        print(f"   ⚠️  Fusion disabled: using pure reranker scores")

    rer_model.eval()

    metrics = {
        'retrieval_acc1': 0,
        'retrieval_acc10': 0,
        'rerank_acc1': 0,
        'rerank_acc10': 0,
        'mrr': 0.0,
        'counted': 0
    }

    sep_token = tokenizer.sep_token if tokenizer.sep_token else "[SEP]"

    num_batches = (len(test_mentions) + batch_size - 1) // batch_size
    pbar = tqdm(range(0, len(test_mentions), batch_size), desc="Reranking batches", total=num_batches)

    for batch_start in pbar:
        batch_end = min(batch_start + batch_size, len(test_mentions))
        batch_mentions = test_mentions[batch_start:batch_end]
        batch_queries = test_queries[batch_start:batch_end]
        batch_candidates = candidates_per_query[batch_start:batch_end]
        batch_ret_scores = retrieval_scores[batch_start:batch_end]

        batch_pairs, batch_metadata = [], []

        for i, m in enumerate(batch_mentions):
            gold = m["mesh"]
            cand_aliases, cand_cids = batch_candidates[i]
            ret_scores = batch_ret_scores[i]

            gold_found_in_top10 = any(check_mesh_match(cand_id, gold) for cand_id in cand_cids)
            gold_is_rank1 = check_mesh_match(cand_cids[0], gold) if cand_cids else False

            metrics['counted'] += 1

            if gold_is_rank1:
                metrics['retrieval_acc1'] += 1
            if gold_found_in_top10:
                metrics['retrieval_acc10'] += 1

            if gold_found_in_top10:
                query_text = batch_queries[i]
                pairs = [f"{query_text} {sep_token} {a}" for a in cand_aliases]

                start_idx = len(batch_pairs)
                batch_pairs.extend(pairs)
                end_idx = len(batch_pairs)

                batch_metadata.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'gold': gold,
                    'cand_cids': cand_cids,
                    'ret_scores': ret_scores  # NEW: Store retrieval scores
                })

        if not batch_pairs:
            continue

        inputs = tokenizer(batch_pairs, padding=True, truncation=True,
                           max_length=max_length, return_tensors="pt")
        model_device = next(rer_model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            all_logits = rer_model(**inputs).logits.squeeze(-1)

        for meta in batch_metadata:
            query_logits = all_logits[meta['start_idx']:meta['end_idx']].cpu()
            ret_scores_tensor = torch.tensor(meta['ret_scores'], dtype=torch.float32)

            # NEW: Score fusion logic
            if use_fusion and len(meta['ret_scores']) > 0 and max(meta['ret_scores']) > 0:
                # Normalize both scores to [0, 1] range
                ret_norm = (ret_scores_tensor - ret_scores_tensor.min()) / (
                            ret_scores_tensor.max() - ret_scores_tensor.min() + 1e-8)
                rerank_norm = (query_logits - query_logits.min()) / (query_logits.max() - query_logits.min() + 1e-8)

                # Fused score: alpha * retrieval + (1-alpha) * reranker
                fused_scores = fusion_alpha * ret_norm + (1 - fusion_alpha) * rerank_norm
                ranked_indices = torch.argsort(fused_scores, descending=True).cpu().tolist()
            else:
                # Pure reranker scores (original behavior)
                ranked_indices = torch.argsort(query_logits, descending=True).cpu().tolist()

            ranked_cids = [meta['cand_cids'][idx] for idx in ranked_indices]

            if check_mesh_match(ranked_cids[0], meta['gold']):
                metrics['rerank_acc1'] += 1

            if any(check_mesh_match(cid, meta['gold']) for cid in ranked_cids[:10]):
                metrics['rerank_acc10'] += 1

            for rank_idx, cid in enumerate(ranked_cids):
                if check_mesh_match(cid, meta['gold']):
                    metrics['mrr'] += 1.0 / (rank_idx + 1)
                    break

        processed = batch_end
        pbar.set_postfix({
            'ret_acc10': f"{metrics['retrieval_acc10']}/{processed}",
            'rer_acc1': f"{metrics['rerank_acc1']}/{processed}"
        })

    total_mentions = len(test_mentions)

    return {
        'retrieval_acc1': metrics['retrieval_acc1'] / total_mentions,
        'retrieval_acc10': metrics['retrieval_acc10'] / total_mentions,
        'rerank_acc1': metrics['rerank_acc1'] / total_mentions,
        'rerank_acc10': metrics['rerank_acc10'] / total_mentions,
        'mrr': metrics['mrr'] / total_mentions,
        'evaluated_on': f"{metrics['counted']}/{total_mentions}",
        'gold_in_top10': metrics['retrieval_acc10']
    }


# ------------------------
# NEW: Alpha Tuning on Validation Set
# ------------------------

def tune_fusion_alpha(rer_model, tokenizer, val_mentions, val_queries,
                      val_candidates, val_scores, args, device):
    """
    Find the best fusion_alpha on validation set
    Tests: [0.5, 0.7, 0.8, 0.9, 0.95]
    """
    print("\n" + "=" * 70)
    print("🔧 TUNING FUSION ALPHA ON VALIDATION SET")
    print("=" * 70)

    alphas_to_test = [0.5, 0.7, 0.8, 0.9, 0.95]
    best_alpha = 0.8
    best_acc1 = -1.0

    results_table = []

    for alpha in alphas_to_test:
        print(f"\n📊 Testing α={alpha:.2f}...")
        results = evaluate_reranker_with_fusion(
            rer_model, tokenizer, val_mentions, val_queries,
            val_candidates, val_scores,
            args.max_length, device, batch_size=args.eval_batch_size,
            fusion_alpha=alpha, use_fusion=True
        )

        acc1 = results['rerank_acc1']
        results_table.append({
            'alpha': alpha,
            'acc1': acc1,
            'acc10': results['rerank_acc10'],
            'mrr': results['mrr']
        })

        print(f"   Rerank Acc@1: {acc1 * 100:.2f}%")

        if acc1 > best_acc1:
            best_acc1 = acc1
            best_alpha = alpha

    print("\n" + "=" * 70)
    print("ALPHA TUNING RESULTS")
    print("=" * 70)
    print(f"{'Alpha':>8} {'Acc@1':>10} {'Acc@10':>10} {'MRR':>10}")
    print("-" * 70)
    for r in results_table:
        marker = " ⭐" if r['alpha'] == best_alpha else ""
        print(f"{r['alpha']:>8.2f} {r['acc1'] * 100:>9.2f}% {r['acc10'] * 100:>9.2f}% {r['mrr']:>10.4f}{marker}")
    print("=" * 70)
    print(f"\n✨ Best α: {best_alpha:.2f} (Acc@1: {best_acc1 * 100:.2f}%)")

    return best_alpha


# ------------------------
# Evaluation Runner
# ------------------------

def run_eval_on_split(split_name: str,
                      ds: Dict,
                      searcher: SAPBERTIndexSearcher,
                      tokenizer: AutoTokenizer,
                      rer_model: AutoAdapterModel,
                      args,
                      device,
                      category: str = "Both",
                      fusion_alpha: float = 0.8):
    if split_name not in ds:
        print(f"\n⚠️  Split '{split_name}' not found. Available: {list(ds.keys())}")
        return None

    print("\n" + "=" * 70)
    header = f"[EVAL] {split_name.upper()} SPLIT (TOP-10, CATEGORY: {category.upper()})"
    if args.use_fusion:
        header += f" [FUSION α={fusion_alpha:.2f}]"
    print(header.center(70))
    print("=" * 70)

    mentions = collect_mentions(ds[split_name], split_name, category)
    if not mentions:
        print(f"  ⚠️  No mentions in {split_name} split — skipping.")
        return None

    retrieval_queries = [to_retrieval_query(m) for m in
                         tqdm(mentions, desc=f"Building {split_name} retrieval queries", leave=False)]

    if args.query_mode == "context":
        rerank_queries = [to_rerank_query(m, window_chars=args.context_window_chars) for m in
                          tqdm(mentions, desc=f"Building {split_name} rerank queries", leave=False)]
    else:
        rerank_queries = [to_retrieval_query(m) for m in mentions]

    print(f"🔍 Retrieving top-10 candidates using SAPBERTIndexSearcher...")
    search_results = searcher.batch_search(retrieval_queries, k=10)
    candidates, ret_scores = extract_candidates_and_scores(search_results)

    # Check if we have valid scores
    has_scores = any(max(scores) > 0 if scores else False for scores in ret_scores)
    if args.use_fusion and not has_scores:
        print(f"  ⚠️  WARNING: No retrieval scores found in search results!")
        print(f"  ⚠️  Fusion will be disabled for this evaluation.")
        print(f"  ℹ️  To enable fusion, ensure SAPBERTIndexSearcher returns 'score' or 'similarity' in results.")
        use_fusion_actual = False
    else:
        use_fusion_actual = args.use_fusion

    results = evaluate_reranker_with_fusion(
        rer_model, tokenizer, mentions, rerank_queries, candidates, ret_scores,
        args.max_length, device, batch_size=args.eval_batch_size,
        fusion_alpha=fusion_alpha, use_fusion=use_fusion_actual
    )

    print("\n" + "-" * 70)
    print(f"{split_name.upper()} EVALUATION RESULTS (TOP-10, CATEGORY: {category.upper()})".center(70))
    print("-" * 70)
    print(f"📊 Evaluated on:            {results['evaluated_on']}")
    print(f"🔍 Retrieval Acc@1:         {results['retrieval_acc1'] * 100:>6.2f}%")
    print(f"🔍 Retrieval Acc@10:        {results['retrieval_acc10'] * 100:>6.2f}%")
    print(f"🥇 Reranker Acc@1:          {results['rerank_acc1'] * 100:>6.2f}%")
    print(f"🏅 Reranker Acc@10:         {results['rerank_acc10'] * 100:>6.2f}%")
    print(f"📈 Mean Reciprocal Rank:    {results['mrr']:>6.4f}")
    if use_fusion_actual:
        print(f"🔀 Fusion α:                {fusion_alpha:.2f}")
    print("-" * 70 + "\n")

    return results


# ------------------------
# Validation Callback
# ------------------------

class ValidationCallback(TrainerCallback):
    """Callback to evaluate on validation set after each epoch and save best model"""

    def __init__(self, val_mentions, val_queries, val_candidates, val_scores, searcher, tokenizer,
                 args, device, ds, output_dir, category="Both", csv_path=None, fusion_alpha=0.8):
        self.val_mentions = val_mentions
        self.val_queries = val_queries
        self.val_candidates = val_candidates
        self.val_scores = val_scores
        self.searcher = searcher
        self.tokenizer = tokenizer
        self.args = args
        self.device = device
        self.ds = ds
        self.output_dir = output_dir
        self.category = category
        self.best_val_rerank_top1 = -1.0
        self.best_epoch = 0
        self.csv_path = csv_path if csv_path else os.path.join(output_dir, "training_results.csv")
        self.fusion_alpha = fusion_alpha

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Evaluate on validation set at the end of each epoch"""
        epoch = int(state.epoch)
        print(f"\n{'=' * 70}")
        print(f"🔎 VALIDATION AFTER EPOCH {epoch}".center(70))
        print(f"{'=' * 70}")

        # Check if we have valid scores
        has_scores = any(max(scores) > 0 if scores else False for scores in self.val_scores)
        use_fusion_actual = self.args.use_fusion and has_scores

        results = evaluate_reranker_with_fusion(
            model, self.tokenizer, self.val_mentions, self.val_queries,
            self.val_candidates, self.val_scores, self.args.max_length, self.device,
            batch_size=self.args.eval_batch_size,
            fusion_alpha=self.fusion_alpha, use_fusion=use_fusion_actual
        )

        val_rerank_acc1 = results['rerank_acc1']

        print(f"\n📊 Validation Results (Epoch {epoch}):")
        print(f"  🔍 Retrieval Acc@1:         {results['retrieval_acc1'] * 100:>6.2f}%")
        print(f"  🔍 Retrieval Acc@10:        {results['retrieval_acc10'] * 100:>6.2f}%")
        print(f"  🥇 Reranker Acc@1:          {val_rerank_acc1 * 100:>6.2f}%")
        print(f"  🏅 Reranker Acc@10:         {results['rerank_acc10'] * 100:>6.2f}%")
        print(f"  📈 Mean Reciprocal Rank:    {results['mrr']:>6.4f}")
        if use_fusion_actual:
            print(f"  🔀 Fusion α:                {self.fusion_alpha:.2f}")

        validation_results = {f'validation_{self.category.lower()}': results}
        write_results_to_csv(
            validation_results,
            self.csv_path,
            description=f"Training Epoch {epoch} Validation",
            epoch=epoch,
            context_window_chars=self.args.context_window_chars
        )

        if val_rerank_acc1 > self.best_val_rerank_top1:
            self.best_val_rerank_top1 = val_rerank_acc1
            self.best_epoch = epoch

            best_model_dir = os.path.join(self.output_dir, "best_model")
            print(f"\n✨ New best model! Saving to {best_model_dir}")
            print(f"   Best validation rerank_acc1: {val_rerank_acc1 * 100:.2f}% (epoch {epoch})")

            model.save_pretrained(best_model_dir)
            model.save_adapter(best_model_dir, self.args.rerank_adapter_name)
            self.tokenizer.save_pretrained(best_model_dir)

            import json
            metadata = {
                'epoch': epoch,
                'val_rerank_acc1': val_rerank_acc1,
                'val_retrieval_acc1': results['retrieval_acc1'],
                'val_retrieval_acc10': results['retrieval_acc10'],
                'val_rerank_acc10': results['rerank_acc10'],
                'val_mrr': results['mrr'],
                'category': self.category,
                'use_fusion': use_fusion_actual,
                'fusion_alpha': self.fusion_alpha if use_fusion_actual else None
            }
            with open(os.path.join(best_model_dir, 'best_model_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
        else:
            print(
                f"\n   Current: {val_rerank_acc1 * 100:.2f}% | Best: {self.best_val_rerank_top1 * 100:.2f}% (epoch {self.best_epoch})")

        print(f"{'=' * 70}\n")

        return control


# ------------------------
# MAIN
# ------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not HAS_FAISS:
        raise RuntimeError("FAISS not available. Install with: pip install faiss-cpu (or faiss-gpu)")

    args.k = 10

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("\n" + "=" * 70)
    print("BC5CDR RERANKER TRAINER WITH SCORE FUSION".center(70))
    print("=" * 70)
    print(f"🖥️  Device: {device}")
    print(f"🌱 Seed: {args.seed}")
    print(f"🎯 Candidate set size (k): 10")
    if args.use_fusion:
        if args.tune_alpha:
            print(f"🔀 Fusion: ENABLED (auto-tune α on validation)")
        else:
            print(f"🔀 Fusion: ENABLED (α={args.fusion_alpha:.2f})")
    else:
        print(f"🔀 Fusion: DISABLED (pure reranker)")
    print()

    # STEP 1: Load dataset
    print("\n" + "=" * 70)
    print("[STEP 1/7] LOAD BC5CDR DATASET")
    print("=" * 70)
    ds = load_bc5cdr(pubtator_path=args.pubtator_path)

    # STEP 2: Initialize searcher
    print("\n" + "=" * 70)
    print("[STEP 2/7] INITIALIZE SAPBERT INDEX SEARCHER")
    print("=" * 70)

    searcher = SAPBERTIndexSearcher(
        model_name=args.base_model,
        adapter_name=args.retriever_adapter_name,
        adapter_path=args.retriever_adapter_path
    )
    searcher.load_index(args.faiss_index_path)
    stats = searcher.get_index_stats()
    print(f"  ✅ Entities: {stats.get('num_entities', '?'):,}")
    print(f"  ✅ Index type: {stats.get('index_type', '?')}")
    print(f"  ✅ Adapter: {'ON' if stats.get('use_adapter') else 'OFF'}")

    searcher._load_model()

    # STEP 3: Prepare training data
    print("\n" + "=" * 70)
    print(f"[STEP 3/7] PREPARE {args.train_split.upper()} SPLIT FOR TRAINING")
    print("=" * 70)

    train_mentions = collect_mentions(ds[args.train_split], args.train_split, args.category)
    print(f"  ✅ Collected {len(train_mentions):,} mentions")

    print(f"🔄 Building queries (mode: {args.query_mode})...")
    train_retrieval_queries = [to_retrieval_query(m) for m in
                               tqdm(train_mentions, desc="Building retrieval queries", leave=False)]

    if args.query_mode == "context":
        train_rerank_queries = [to_rerank_query(m, window_chars=args.context_window_chars) for m in
                                tqdm(train_mentions, desc="Building rerank queries", leave=False)]
    else:
        train_rerank_queries = [to_retrieval_query(m) for m in train_mentions]

    print(f"🔍 Retrieving top-10 candidates using SAPBERTIndexSearcher...")
    train_search_results = searcher.batch_search(train_retrieval_queries, k=10)
    train_candidates, train_scores = extract_candidates_and_scores(train_search_results)

    # Check if scores are available
    has_train_scores = any(max(scores) > 0 if scores else False for scores in train_scores)
    if args.use_fusion and not has_train_scores:
        print(f"  ⚠️  WARNING: No retrieval scores found in search results!")
        print(
            f"  ℹ️  Sample result keys: {list(train_search_results[0][0].keys()) if train_search_results and train_search_results[0] else 'N/A'}")
        print(f"  ℹ️  Fusion will be disabled during evaluation.")

    print(f"🔗 Building training groups (gold + top-9 negatives from top-10)...")
    pairs: List[Tuple[str, List[str]]] = []
    kept = 0
    forced_gold = 0

    gold_at_rank1 = 0
    gold_in_top10_not_rank1 = 0
    gold_not_in_top10 = 0

    for i in tqdm(range(len(train_mentions)), desc="Creating pairs", leave=False):
        m = train_mentions[i]
        gold = m["mesh"]
        cand_aliases, cand_cids = train_candidates[i]

        gold_found = False
        gold_idx = -1
        for idx, cand_id in enumerate(cand_cids):
            if check_mesh_match(cand_id, gold):
                gold_found = True
                gold_idx = idx
                break

        if not gold_found:
            gold_not_in_top10 += 1
            gold_alias = m["text"]
            negs = cand_aliases[:9] if len(cand_aliases) >= 9 else cand_aliases
            forced_gold += 1
        else:
            if gold_idx == 0:
                gold_at_rank1 += 1
            else:
                gold_in_top10_not_rank1 += 1

            gold_alias = cand_aliases[gold_idx]
            negs = [a for j, (a, c) in enumerate(zip(cand_aliases, cand_cids)) if j != gold_idx]

        cand_list = [gold_alias] + negs
        pairs.append((train_rerank_queries[i], cand_list))
        kept += 1

    total_mentions = len(train_mentions)

    print(f"\n📊 Training Data Analysis:")
    print(f"  {'=' * 60}")
    print(f"  Total mentions:                    {total_mentions:>8,}")
    print(f"  {'=' * 60}")
    print(f"  ✅ FAISS correct @1:               {gold_at_rank1:>8,} ({gold_at_rank1 / total_mentions * 100:>5.2f}%)")
    print(f"     (Already perfect - no improvement needed)")
    print(f"  ")
    print(
        f"  🎯 FAISS wrong @1, gold in top-10: {gold_in_top10_not_rank1:>8,} ({gold_in_top10_not_rank1 / total_mentions * 100:>5.2f}%)")
    print(f"     ⭐ THIS IS WHAT RERANKING CAN IMPROVE!")
    print(f"  ")
    print(
        f"  ❌ Gold not in top-10:             {gold_not_in_top10:>8,} ({gold_not_in_top10 / total_mentions * 100:>5.2f}%)")
    print(f"     (Cannot improve with reranking)")
    print(f"  {'=' * 60}")
    print(f"  Total:                             {total_mentions:>8,} (100.00%)")
    print(f"  {'=' * 60}")
    print(f"\n💡 Key Insight:")
    print(f"   Maximum possible improvement from reranking: {gold_in_top10_not_rank1 / total_mentions * 100:.2f}%")
    print(f"   (If reranker can perfectly move all rank 2-10 to rank 1)\n")

    print(f"  ✅ Created {kept:,} training groups (100% of mentions)")

    # STEP 4: Build reranker
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

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if args.query_mode == "context":
        special_tokens = {"additional_special_tokens": ["[MENTION]", "[/MENTION]"]}
        num_added = tok.add_special_tokens(special_tokens)
        if num_added > 0:
            print(f"  ✅ Added {num_added} special tokens: [MENTION], [/MENTION]")
            rer_model.resize_token_embeddings(len(tok))
            print(f"  ✅ Resized model embeddings to {len(tok)}")

    train_ds = make_listwise_dataset(tok, pairs, max_length=args.max_length)

    if HAS_WANDB:
        run_name = f"bc5cdr_rerank_{args.category}_{args.query_mode}_lr{args.lr}_e{args.epochs}"
        if args.use_fusion:
            run_name += f"_fusion{args.fusion_alpha:.2f}"

        wandb.init(
            project="bc5cdr-reranker",
            name=run_name,
            config={
                "base_model": args.base_model,
                "query_mode": args.query_mode,
                "context_window_chars": args.context_window_chars,
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "batch_size": args.per_device_train_batch_size,
                "max_length": args.max_length,
                "train_split": args.train_split,
                "category": args.category,
                "seed": args.seed,
                "weight_decay": args.weight_decay,
                "use_fusion": args.use_fusion,
                "fusion_alpha": args.fusion_alpha if args.use_fusion else None,
            },
            resume="allow",
            id=run_name.replace(".", "_").replace("/", "_"),
        )
        print(f"  ✅ W&B initialized: project=bc5cdr-reranker, run={run_name}")

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
        report_to=["wandb"] if HAS_WANDB else [],
        seed=args.seed,
        use_mps_device=(device.type == "mps")
    )

    # STEP 5: Training
    print("\n" + "=" * 70)
    print("[STEP 5/7] TRAIN RERANKER ADAPTER")
    print("=" * 70)

    grouped_sampler = QueryGroupedSampler(train_ds, groups_per_batch=1)
    collator = QueryGroupedCollator(tok)

    trainer = ListwiseAdapterTrainer(
        model=rer_model,
        args=args_hf,
        train_dataset=train_ds,
        tokenizer=tok,
        grouped_sampler=grouped_sampler,
        data_collator=collator,
    )

    # Prepare validation data
    print("\n" + "=" * 70)
    print("[STEP 5.5/7] PREPARE VALIDATION SPLIT")
    print("=" * 70)

    val_mentions = collect_mentions(ds['validation'], 'validation', args.category)
    print(f"  ✅ Collected {len(val_mentions):,} validation mentions")

    val_retrieval_queries = [to_retrieval_query(m) for m in
                             tqdm(val_mentions, desc="Building validation retrieval queries", leave=False)]

    if args.query_mode == "context":
        val_rerank_queries = [to_rerank_query(m, window_chars=args.context_window_chars) for m in
                              tqdm(val_mentions, desc="Building validation rerank queries", leave=False)]
    else:
        val_rerank_queries = [to_retrieval_query(m) for m in val_mentions]

    print(f"🔍 Retrieving validation candidates...")
    val_search_results = searcher.batch_search(val_retrieval_queries, k=10)
    val_candidates, val_scores = extract_candidates_and_scores(val_search_results)

    # STEP 5.75: Tune alpha if requested
    fusion_alpha = args.fusion_alpha
    if args.tune_alpha and args.use_fusion:
        fusion_alpha = tune_fusion_alpha(
            rer_model, tok, val_mentions, val_rerank_queries,
            val_candidates, val_scores, args, device
        )
        print(f"\n✨ Using tuned α={fusion_alpha:.2f} for training and evaluation")

    validation_callback = ValidationCallback(
        val_mentions=val_mentions,
        val_queries=val_rerank_queries,
        val_candidates=val_candidates,
        val_scores=val_scores,
        searcher=searcher,
        tokenizer=tok,
        args=args,
        device=device,
        ds=ds,
        output_dir=args.output_dir,
        category=args.category,
        csv_path=os.path.join(args.output_dir, "training_results.csv"),
        fusion_alpha=fusion_alpha
    )

    trainer.add_callback(validation_callback)

    if not args.evaluate_only and args.epochs > 0:
        print(f"\n🚀 Starting training ({args.epochs} epochs)...")
        print(f"   Total training groups: {len(pairs)}")
        print(f"   Total training examples (pairs): {len(train_ds)}")
        print(f"   ✅ Validation callback enabled (will evaluate after each epoch)")
        trainer.train()
        print(f"\n💾 Saving final model to: {args.output_dir}")
        trainer.save_model(args.output_dir)
        rer_model.save_adapter(args.output_dir, args.rerank_adapter_name)
        print(f"  ✅ Model saved successfully")

        # Load best model
        best_model_path = os.path.join(args.output_dir, "best_model")
        if os.path.exists(best_model_path):
            print(f"\n🏆 Loading best model from: {best_model_path}")
            print(
                f"   Best validation rerank_top1: {validation_callback.best_val_rerank_top1 * 100:.2f}% (epoch {validation_callback.best_epoch})")
            rer_model = AutoAdapterModel.from_pretrained(args.base_model)
            rer_model.load_adapter(best_model_path, load_as=args.rerank_adapter_name, with_head=True)
            rer_model.set_active_adapters([args.rerank_adapter_name])
            rer_model.to(device)
    else:
        print("  ⏭️  Skipping training (evaluate_only or epochs=0)")

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION PHASE (BEST MODEL)".center(70))
    print("=" * 70)

    final_csv_path = os.path.join(args.output_dir, "final_evaluation_results.csv")
    all_final_results = {}

    print(f"\n📝 Final evaluation results will be saved to: {final_csv_path}")

    if args.category == "Both":
        for split in ["train", "validation", "test"]:
            disease_results = run_eval_on_split(split, ds, searcher, tok, rer_model, args, device,
                                                category="Disease", fusion_alpha=fusion_alpha)
            chemical_results = run_eval_on_split(split, ds, searcher, tok, rer_model, args, device,
                                                 category="Chemical", fusion_alpha=fusion_alpha)
            if disease_results:
                all_final_results[f"{split}_disease"] = disease_results
            if chemical_results:
                all_final_results[f"{split}_chemical"] = chemical_results
    else:
        for split in ["train", "validation", "test"]:
            results = run_eval_on_split(split, ds, searcher, tok, rer_model, args, device,
                                        category=args.category, fusion_alpha=fusion_alpha)
            if results:
                all_final_results[f"{split}_{args.category.lower()}"] = results

    if all_final_results:
        print(f"\n📊 Writing {len(all_final_results)} evaluation result(s) to CSV...")
        best_epoch = validation_callback.best_epoch if not args.evaluate_only and args.epochs > 0 else None
        write_results_to_csv(
            all_final_results,
            final_csv_path,
            description="Best Model Final Evaluation with Fusion" if args.use_fusion else "Best Model Final Evaluation",
            epoch=best_epoch,
            context_window_chars=args.context_window_chars
        )
        print(f"✅ Final evaluation results saved to: {final_csv_path}")

    print("\n" + "=" * 70)
    print("✨ TRAINING AND EVALUATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()

