#!/usr/bin/env python

import argparse
import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple
import logging

import torch
from torch.utils.data import DataLoader
from torch import nn
from transformers import AutoTokenizer, TrainingArguments
from adapters import AutoAdapterModel
from adapters.trainer import AdapterTrainer
from tqdm import tqdm

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # → sapbert/
sys.path.insert(0, str(PROJECT_ROOT))
# IMPORTANT: append directories, not files
# sys.path.append("/Users/lakshankarunathilake/PycharmProjects/sapbert/utils/NEL")
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
    p.add_argument("--per_device_train_batch_size", type=int, default=64, help="Items per batch (candidates). If using grouped sampler, each batch is 1+ whole query groups.")
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
                    current_doc['full_text'] += '\n'
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
                                    'normalized': [{'db_name': 'MESH', 'db_id': mesh_id}]
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

    if 'validation' not in ds and 'train' in ds:
        print("  ⚠️  No validation set found, creating from train split (10%)")

    missing = [s for s in ['train', 'validation', 'test'] if s not in ds]
    if missing:
        raise RuntimeError(f"Missing required splits: {missing}. Available: {list(ds.keys())}")

    return ds


def collect_mentions(ds_split, split_name: str = "split") -> List[dict]:
    mentions = []
    print(f"📋 Collecting mentions from {split_name} split...")

    n = len(ds_split["id"])
    for idx in tqdm(range(n), desc=f"Processing {split_name}", leave=False):
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
    text = mention.get("text") or ""
    doc = mention.get("doc_text")
    if not doc:
        return text
    try:
        start = doc.lower().index(text.lower())
        end = start + len(text)
    except ValueError:
        return text
    left = doc[:start]; right = doc[end:]
    lt = tokenizer.tokenize(left)[-window_tokens:]
    rt = tokenizer.tokenize(right)[:window_tokens]
    lstr = tokenizer.convert_tokens_to_string(lt).strip()
    rstr = tokenizer.convert_tokens_to_string(rt).strip()
    return f"{lstr} [MENTION] {text} [/MENTION] {rstr}".strip()


# ------------------------
# Dataset (list-backed)
# ------------------------

@dataclass
class RerankExample:
    input_ids: List[int]
    attention_mask: List[int]
    query_id: int
    label: int  # 1 for gold, 0 for negatives


class SimpleDataset:
    def __init__(self, data: List[Dict]):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


def make_listwise_dataset(tokenizer: AutoTokenizer, pairs: List[Tuple[str, List[str]]], max_length: int) -> SimpleDataset:
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
    return SimpleDataset(features)


# ------------------------
# Collator + Sampler that keep whole query groups together
# ------------------------

class QueryGroupedCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, items: List[Dict]):
        # items are dicts for multiple query groups (complete per group)
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
            'labels': torch.tensor(labels, dtype=torch.long),  # use 'labels' to be HF-friendly
        }


class QueryGroupedSampler:
    """
    Yields batches of indices that contain whole query groups.
    By default, we put exactly ONE full group per batch (safest for listwise).
    If you want more than one group per batch, set groups_per_batch > 1.
    """
    def __init__(self, dataset: SimpleDataset, groups_per_batch: int = 1):
        # group indices by query_id
        groups = {}
        for idx, item in enumerate(dataset.data):
            qid = item["query_id"]
            groups.setdefault(qid, []).append(idx)
        self.group_list = list(groups.values())
        self.groups_per_batch = max(1, int(groups_per_batch))

    def __iter__(self):
        # shuffle groups each epoch
        import random
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
        # approximate number of batches
        from math import ceil
        return ceil(len(self.group_list) / self.groups_per_batch)


# ------------------------
# Custom Trainer that uses our sampler
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
        # accept both 'labels' and 'label'
        labels = inputs.pop("labels", inputs.pop("label", None))
        if labels is None:
            raise KeyError(f"No label field in inputs. Got keys: {list(inputs.keys())}")
        if "query_id" not in inputs:
            raise KeyError(f"'query_id' missing from inputs. Got keys: {list(inputs.keys())}")

        qids = inputs.pop("query_id")
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)

        if hasattr(qids, "ndim") and qids.ndim > 1: qids = qids.view(-1)
        if hasattr(labels, "ndim") and labels.ndim > 1: labels = labels.view(-1)

        loss_terms = []
        for qid in torch.unique(qids):
            mask = (qids == qid)
            group_logits = logits[mask]
            group_labels = labels[mask]

            # one positive per group expected
            pos_idx = (group_labels == 1).nonzero(as_tuple=True)[0]
            if pos_idx.numel() != 1:
                # skip groups without a single positive
                continue

            log_probs = torch.log_softmax(group_logits, dim=0)
            loss_terms.append(-log_probs[pos_idx[0]])

        if loss_terms:
            loss = torch.stack(loss_terms).mean()
        else:
            # return a zero tensor tied to the graph (not a float)
            loss = logits.sum() * 0.0

        return (loss, outputs) if return_outputs else loss


# ------------------------
# Retrieval and evaluation helpers
# ------------------------

def extract_candidates_from_search_results(search_results: List[List[Dict]]) -> List[Tuple[List[str], List[str]]]:
    candidates = []
    for results in search_results:
        cand_aliases, cand_ids = [], []
        for r in results:
            entity_id = r.get('entity_id')
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
    print(f"📊 Evaluating reranker on {len(test_mentions):,} test mentions...")
    rer_model.eval()

    metrics = {'retrieval_recall': 0, 'top1': 0, 'top5': 0, 'mrr': 0.0, 'counted': 0}

    num_batches = (len(test_mentions) + batch_size - 1) // batch_size
    pbar = tqdm(range(0, len(test_mentions), batch_size), desc="Reranking batches", total=num_batches)

    for batch_start in pbar:
        batch_end = min(batch_start + batch_size, len(test_mentions))
        batch_mentions = test_mentions[batch_start:batch_end]
        batch_queries = test_queries[batch_start:batch_end]
        batch_candidates = candidates_per_query[batch_start:batch_end]

        batch_pairs, batch_metadata = [], []

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


def run_eval_on_split(split_name: str,
                      ds: Dict,
                      searcher: SAPBERTIndexSearcher,
                      tokenizer: AutoTokenizer,
                      rer_model: AutoAdapterModel,
                      args,
                      device):
    if split_name not in ds:
        print(f"\n⚠️  Split '{split_name}' not found. Available: {list(ds.keys())}")
        return None

    print("\n" + "=" * 70)
    print(f"[EVAL] {split_name.upper()} SPLIT".center(70))
    print("=" * 70)

    mentions = collect_mentions(ds[split_name], split_name)
    if not mentions:
        print(f"  ⚠️  No mentions in {split_name} split — skipping.")
        return None

    # Build queries (same logic you used for training)
    def to_query(m):
        if args.query_mode == "context":
            return build_context_query(m, searcher.tokenizer, args.context_window)
        return m["text"]

    queries = [to_query(m) for m in tqdm(mentions, desc=f"Building {split_name} queries", leave=False)]

    print(f"🔍 Retrieving top-{args.k} candidates using SAPBERTIndexSearcher...")
    search_results = searcher.batch_search(queries, k=args.k)
    candidates = extract_candidates_from_search_results(search_results)

    results = evaluate_reranker(
        rer_model, tokenizer, mentions, queries, candidates,
        args.max_length, device, batch_size=args.eval_batch_size
    )

    # Pretty print
    print("\n" + "-" * 70)
    print(f"{split_name.upper()} EVALUATION RESULTS".center(70))
    print("-" * 70)
    print(f"📊 Evaluated on:            {results['evaluated_on']}")
    print(f"🔍 Retrieval Recall@{args.k:>2}:      {results['retrieval_recall@k'] * 100:>6.2f}%")
    print(f"🥇 Reranker Top-1:          {results['rerank_top1'] * 100:>6.2f}%")
    print(f"🏅 Reranker Top-5:          {results['rerank_top5'] * 100:>6.2f}%")
    print(f"📈 Mean Reciprocal Rank:    {results['mrr']:>6.4f}")
    print("-" * 70 + "\n")

    return results


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if not HAS_FAISS:
        raise RuntimeError("FAISS not available. Install with: pip install faiss-cpu (or faiss-gpu)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70)
    print("BC5CDR RERANKER TRAINER (using SAPBERTIndexSearcher)".center(70))
    print("=" * 70)
    print(f"🖥️  Device: {device}")
    print(f"🌱 Seed: {args.seed}")
    print()

    # STEP 1
    print("\n" + "=" * 70)
    print("[STEP 1/7] LOAD BC5CDR DATASET")
    print("=" * 70)
    ds = load_bc5cdr(pubtator_path=args.pubtator_path)
    print(f"  ✅ Train docs: {len(ds['train']['id'])}")
    print(f"  ✅ Validation docs: {len(ds['validation']['id'])}")
    print(f"  ✅ Test docs: {len(ds['test']['id'])}")

    def count_entity_rows(ds_split):
        # Count raw entity rows (before splitting on multiple mesh ids)
        n_docs = len(ds_split["id"])
        rows = 0
        for ents in ds_split["entities"]:
            rows += len(ents)
        return n_docs, rows

    n_docs, raw_rows = count_entity_rows(ds["test"])
    print(f"[DEBUG] Test docs: {n_docs}, raw entity rows: {raw_rows}")

    test_mentions = collect_mentions(ds["test"], "test")
    print(f"[DEBUG] Mentions built (after splitting by multiple MeSH IDs): {len(test_mentions)}")

    # STEP 2
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
    print(f"  ✅ Entities: {stats.get('num_entities','?'):,}")
    print(f"  ✅ Index type: {stats.get('index_type','?')}")
    print(f"  ✅ Adapter: {'ON' if stats.get('use_adapter') else 'OFF'}")

    # STEP 3
    print("\n" + "=" * 70)
    print(f"[STEP 3/7] PREPARE {args.train_split.upper()} SPLIT FOR TRAINING")
    print("=" * 70)

    train_mentions = collect_mentions(ds[args.train_split], args.train_split)
    print(f"  ✅ Collected {len(train_mentions):,} mentions")

    searcher._load_model()  # ensure tokenizer available

    def to_query(m):
        if args.query_mode == "context":
            return build_context_query(m, searcher.tokenizer, args.context_window)
        return m["text"]

    print(f"🔄 Building queries (mode: {args.query_mode})...")
    train_queries = [to_query(m) for m in tqdm(train_mentions, desc="Building queries", leave=False)]

    print(f"🔍 Retrieving top-{args.k} candidates using SAPBERTIndexSearcher...")
    train_search_results = searcher.batch_search(train_queries, k=args.k)  # no 'desc' param
    train_candidates = extract_candidates_from_search_results(train_search_results)

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
    print(f"  ✅ Training recall@{args.k}: {kept / max(1,len(train_mentions)) * 100:.2f}%")

    # STEP 4
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
        per_device_train_batch_size=args.per_device_train_batch_size,  # not used by our batch_sampler, but fine
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        weight_decay=args.weight_decay,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="no",
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
        use_mps_device=False,
        use_cpu=True
    )

    # STEP 5
    print("\n" + "=" * 70)
    print("[STEP 5/7] TRAIN RERANKER ADAPTER")
    print("=" * 70)

    # ONE WHOLE QUERY GROUP PER BATCH (safest)
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

    if not args.evaluate_only and args.epochs > 0:
        print(f"🚀 Starting training ({args.epochs} epochs)...")
        print(f"   Groups per batch: 1 (one query per batch)")
        print(f"   Total training groups (queries): {len(pairs)}")
        print(f"   Total training examples (pairs): {len(train_ds)}")
        trainer.train()
        print(f"\n💾 Saving model to: {args.output_dir}")
        trainer.save_model(args.output_dir)
        rer_model.save_adapter(args.output_dir, args.rerank_adapter_name)
        print(f"  ✅ Model saved successfully")
    else:
        print("  ⏭️  Skipping training (evaluate_only or epochs=0)")

    # EVALUATE on VALIDATION (dev) split
    run_eval_on_split("train", ds, searcher, tok, rer_model, args, device)

    # EVALUATE on VALIDATION (dev) split
    run_eval_on_split("validation", ds, searcher, tok, rer_model, args, device)

    # EVALUATE on TEST split
    run_eval_on_split("test", ds, searcher, tok, rer_model, args, device)


if __name__ == "__main__":
    main()
