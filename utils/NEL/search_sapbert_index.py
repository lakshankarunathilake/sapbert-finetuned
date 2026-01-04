#!/usr/bin/env python3
"""
SAPBERT Index Searcher (Adapter-aware)

Search and query existing SAPBERT FAISS indexes for entity similarity.

Usage:
    python search_sapbert_index.py --index_path ./indexes/wikidata_index --query "diabetes"

Examples:
    # Use adapter settings saved in the index (recommended)
    python search_sapbert_index.py --index_path ./indexes/wikidata_index --query "heart disease"

    # Override to use an adapter from local path
    python search_sapbert_index.py --index_path ./indexes/wikidata_index --adapter_path ./my_adapter --query "heart disease"

    # Or load adapter from the Hub
    python search_sapbert_index.py --index_path ./indexes/wikidata_index --adapter_name username/my-adapter --query "heart disease"
"""

import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import pickle
import json
import argparse
from typing import List, Dict, Optional
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SAPBERTIndexSearcher:
    """
    Utility class for searching SAPBERT FAISS indexes (with optional adapter support)
    """

    def __init__(self,
                 model_name: Optional[str] = None,
                 adapter_name: Optional[str] = None,
                 adapter_path: Optional[str] = None):
        """
        Initialize the searcher

        Args:
            model_name: HF model name for SAPBERT (loaded from index config if None)
            adapter_name: Name of the adapter on HF Hub (overrides index config if provided)
            adapter_path: Local path to adapter weights (overrides index config if provided)
        """
        self.model_name = model_name
        self.cli_adapter_name = adapter_name
        self.cli_adapter_path = adapter_path
        self.use_adapter = False  # will be set after loading index/config and applying overrides

        # Mac-specific device selection
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                self.device = torch.device('mps')
                _ = torch.randn(1, 1).to(self.device)  # smoke test
                logger.info("MPS (Metal Performance Shaders) is available and working")
            except Exception as e:
                logger.warning(f"MPS failed test, falling back to CPU: {e}")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')

        self.tokenizer = None
        self.model = None
        self.index = None
        self.metadata = {}
        self.embedding_dim = 768
        self.index_config: Dict = {}

        # Resolved (effective) adapter settings after load_index()
        self.adapter_name: Optional[str] = None
        self.adapter_path: Optional[str] = None

        logger.info(f"Initialized SAPBERTIndexSearcher with device: {self.device}")

        # Set multiprocessing method for Mac compatibility
        import multiprocessing as mp
        try:
            if mp.get_start_method(allow_none=True) != 'spawn':
                mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    # ✅ FIX 1: metadata compatibility helper
    def _iter_metadata(self):
        """Iterate over metadata entries regardless of list/dict format."""
        if isinstance(self.metadata, dict):
            return self.metadata.values()
        elif isinstance(self.metadata, list):
            return self.metadata
        else:
            return []

    def _resolve_adapter_settings(self):
        """
        Resolve final adapter settings based on:
          1) CLI overrides (highest priority)
          2) Index config (fallback)
        """
        cfg_use_adapter = bool(self.index_config.get('use_adapter', False))
        cfg_adapter_name = self.index_config.get('adapter_name')
        cfg_adapter_path = self.index_config.get('adapter_path')

        # Apply overrides (mutually exclusive)
        if self.cli_adapter_name and self.cli_adapter_path:
            raise ValueError("Cannot specify both --adapter_name and --adapter_path. Choose one.")

        if self.cli_adapter_name is not None or self.cli_adapter_path is not None:
            # Explicit override turns adapter mode on
            self.use_adapter = True
            self.adapter_name = self.cli_adapter_name
            self.adapter_path = self.cli_adapter_path
        else:
            # Follow index config
            self.use_adapter = cfg_use_adapter
            self.adapter_name = cfg_adapter_name
            self.adapter_path = cfg_adapter_path

        # Minor sanity/logging
        if self.use_adapter:
            src = "CLI override" if (self.cli_adapter_name or self.cli_adapter_path) else "index config"
            if self.adapter_path:
                logger.info(f"Adapter enabled ({src}), using local path: {self.adapter_path}")
                if not os.path.exists(self.adapter_path):
                    logger.warning(f"Adapter path does not exist: {self.adapter_path}")
            elif self.adapter_name:
                logger.info(f"Adapter enabled ({src}), using Hub name: {self.adapter_name}")
            else:
                logger.warning("Adapter mode requested but neither adapter_name nor adapter_path was provided. Disabling.")
                self.use_adapter = False

    def _load_model(self):
        """Load SAPBERT model + tokenizer (and adapter if requested)"""
        if self.model is not None and self.tokenizer is not None:
            return

        model_name = self.model_name or self.index_config.get('model_name',
                                                              'cambridgeltl/SapBERT-from-PubMedBERT-fulltext')

        logger.info(f"Loading SAPBERT tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.use_adapter:
            try:
                from adapters import AutoAdapterModel
            except ImportError as e:
                logger.error("adapters library not found. Install with: pip install adapters")
                raise

            logger.info("Loading model with adapters library...")
            self.model = AutoAdapterModel.from_pretrained(model_name)

            # Load adapter (path wins over name)
            try:
                if self.adapter_path:
                    logger.info(f"Loading adapter from local path: {self.adapter_path}")
                    active_name = self.model.load_adapter(self.adapter_path, source="local")
                else:
                    logger.info(f"Loading adapter from Hub: {self.adapter_name}")
                    active_name = self.model.load_adapter(self.adapter_name)

                self.model.set_active_adapters(active_name)
                logger.info(f"Adapter '{active_name}' activated")

                # ✅ FIX 2: Disable heads safely (version-safe)
                if hasattr(self.model, "set_active_head"):
                    try:
                        self.model.set_active_head(None)
                    except Exception:
                        pass
                elif hasattr(self.model, "active_head"):
                    try:
                        self.model.active_head = None
                    except Exception:
                        pass

            except Exception as e:
                logger.error(f"Failed to load/activate adapter: {e}")
                raise
        else:
            logger.info(f"Loading base model without adapters: {model_name}")
            self.model = AutoModel.from_pretrained(model_name)

        # Device + eval + no-grad
        try:
            self.model.to(self.device)
        except Exception as e:
            logger.warning(f"Failed to move model to {self.device}, using CPU: {e}")
            self.device = torch.device('cpu')
            self.model.to(self.device)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        logger.info(f"Model loaded successfully on {self.device} (adapter={'on' if self.use_adapter else 'off'})")

    def load_index(self, index_path: str) -> Dict:
        """
        Load FAISS index + metadata + config
        """
        try:
            self.index = faiss.read_index(f"{index_path}.faiss")
            with open(f"{index_path}_metadata.pkl", 'rb') as f:
                self.metadata = pickle.load(f)
            with open(f"{index_path}_config.json", 'r') as f:
                self.index_config = json.load(f)

            self.embedding_dim = self.index_config.get('embedding_dim', 768)

            # ✅ FIX 3: dimension safety
            if hasattr(self.index, "d") and self.index.d != self.embedding_dim:
                logger.warning(
                    f"Index dimension ({self.index.d}) != config embedding_dim ({self.embedding_dim}). "
                    f"Using index.d={self.index.d}"
                )
                self.embedding_dim = self.index.d

            # ✅ FIX 4: warn if index not normalized (important for cosine similarity)
            if not self.index_config.get("normalized", False):
                logger.warning(
                    "Index config does not confirm embeddings are normalized. "
                    "Cosine similarity via inner product assumes normalized embeddings."
                )

            # Update model name from config if not specified
            if self.model_name is None:
                self.model_name = self.index_config.get('model_name')

            # Resolve final adapter settings
            self._resolve_adapter_settings()

            logger.info(f"Loaded index from {index_path}")
            logger.info(f"Index contains {len(self.metadata)} entities")
            logger.info(f"Index type: {self.index_config.get('index_type', 'Unknown')}")
            return self.index_config
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise

    def _generate_single_embedding(self, text: str, max_length: int = 16) -> np.ndarray:
        """Generate embedding for a single text using efficient tokenization"""
        try:
            toks = self.tokenizer.batch_encode_plus(
                [text],
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=True
            )
            try:
                toks_device = {k: v.to(self.device) for k, v in toks.items()}
            except Exception as e:
                logger.warning(f"Failed to move inputs to {self.device}, using CPU: {e}")
                self.device = torch.device('cpu')
                self.model.to(self.device)
                toks_device = {k: v.to(self.device) for k, v in toks.items()}

            with torch.no_grad():
                outputs = self.model(**toks_device)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                if self.device.type in ['cuda', 'mps']:
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
            return embedding.squeeze()
        except Exception as e:
            logger.error(f"Error generating embedding for text: {text[:50]}... Error: {e}")
            raise

    def search(self,
               query_text: str,
               k: int = 10,
               return_scores: bool = True) -> List[Dict]:
        """
        Search for similar entities
        """
        if self.index is None:
            raise ValueError("Index not loaded! Use load_index() first.")

        self._load_model()

        # Keep tokenization consistent with index creation
        max_length = self.index_config.get('max_length', 16)
        query_embedding = self._generate_single_embedding(query_text, max_length)

        # Normalize for cosine similarity in IP space
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, k)

        results = []
        seen_entities = set()

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            # ✅ FIX 5: support both list/dict metadata formats safely
            if isinstance(self.metadata, dict):
                result = self.metadata[idx].copy()
            else:
                result = self.metadata[idx].copy()

            eid = result['entity_id']
            if eid in seen_entities:
                continue
            seen_entities.add(eid)

            if return_scores:
                result['similarity_score'] = float(score)
            if 'all_aliases' in result:
                result['aliases'] = result['all_aliases']
            results.append(result)

        if return_scores and results:
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
        for i, r in enumerate(results):
            r['rank'] = i + 1
        return results

    def batch_search(self, queries: List[str], k: int = 10) -> List[List[Dict]]:
        return [self.search(q, k) for q in tqdm(queries, desc="Processing queries")]

    # ✅ FIX 6: get_entity_by_id works for both list/dict and avoids mutating metadata
    def get_entity_by_id(self, entity_id: str) -> Optional[Dict]:
        for md in self._iter_metadata():
            if md.get('entity_id') == entity_id:
                out = md.copy()
                if 'all_aliases' in out:
                    out['aliases'] = out['all_aliases']
                return out
        return None

    def get_index_stats(self) -> Dict:
        if self.index is None:
            return {"error": "No index loaded"}
        return {
            "num_entities": len(self.metadata),
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_config.get('index_type', 'Unknown'),
            "model_name": self.index_config.get('model_name', 'Unknown'),
            "is_trained": getattr(self.index, 'is_trained', True),
            "created_at": self.index_config.get('created_at', 'Unknown'),
            "processing_time_minutes": self.index_config.get('processing_time_minutes', 'Unknown'),
            "max_length": self.index_config.get('max_length', 'Unknown'),
            "use_adapter": self.use_adapter,
            "adapter_name": self.adapter_name,
            "adapter_path": self.adapter_path,
        }

    # ✅ FIX 7: move get_stats into class (was previously outside and unusable)
    def get_stats(self) -> Dict:
        """
        Backwards-compatible alias for get_index_stats().
        Returns keys used by downstream code:
          num_entities, index_type, model_name, use_adapter, adapter_name, adapter_path, ...
        """
        return self.get_index_stats()

    def get_similar_entities(self, entity_id: str, k: int = 10) -> List[Dict]:
        ent = self.get_entity_by_id(entity_id)
        if ent is None:
            return []
        results = self.search(ent['processed_text'], k + 1)
        return [r for r in results if r['entity_id'] != entity_id][:k]

    def export_search_results(self, results: List[Dict], output_file: str = "search_results.json"):
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Search results exported to {output_file}")

    def debug_embedding(self, text: str) -> Dict:
        self._load_model()
        max_length = self.index_config.get('max_length', 16)
        emb = self._generate_single_embedding(text, max_length)
        return {
            'text': text,
            'max_length_used': max_length,
            'embedding_shape': emb.shape,
            'embedding_norm': np.linalg.norm(emb),
            'embedding_mean': np.mean(emb),
            'embedding_std': np.std(emb),
            'has_nan': np.isnan(emb).any(),
            'has_inf': np.isinf(emb).any(),
            'is_zero': np.allclose(emb, 0),
            'min_value': np.min(emb),
            'max_value': np.max(emb)
        }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Search SAPBERT FAISS index for entity similarity (adapter-aware)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use adapter settings saved with the index
  python search_sapbert_index.py --index_path ./my_indexes/biomedical_entities --query "diabetes"

  # Override adapter (Hub)
  python search_sapbert_index.py --index_path ./my_indexes/biomedical_entities --adapter_name username/my-adapter --query "diabetes"

  # Override adapter (local path)
  python search_sapbert_index.py --index_path ./my_indexes/biomedical_entities --adapter_path ./my_adapter --query "diabetes"
        """
    )

    parser.add_argument('--mode', type=str,
                        choices=['search', 'batch_search', 'get_entity', 'similar', 'stats', 'debug'],
                        default='search', help='Search mode (default: search)')

    parser.add_argument('--index_path', type=str, default='./indexes/entities_index',
                        help='Path to index files without extension')

    parser.add_argument('--query', type=str, default='diabetes mellitus',
                        help='Search query text')

    parser.add_argument('--queries', type=str, default='diabetes,cancer,heart disease',
                        help='Comma-separated queries for batch search')

    parser.add_argument('--entity_id', type=str, default='Q12206',
                        help='Entity ID for lookup/similar')

    parser.add_argument('--k', type=int, default=10, help='Number of results to return')

    parser.add_argument('--export', type=str, help='Export results to JSON file')

    parser.add_argument('--adapter_name', type=str, default=None,
                        help='Adapter name from HuggingFace Hub (overrides index config)')
    parser.add_argument('--adapter_path', type=str, default=None,
                        help='Local path to adapter weights (overrides index config)')

    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    return parser.parse_args()


def search_mode(searcher, args):
    logger.info(f"Searching for: '{args.query}' (k={args.k})")
    results = searcher.search(args.query, k=args.k)
    print(f"\n🔍 Search Results for: '{args.query}'")
    print("=" * 80)
    if not results:
        print("No results found.")
        return results
    for result in results:
        print(f"Rank {result['rank']}: {result['entity_id']}")
        print(f"  Score: {result['similarity_score']:.4f}")
        if 'primary_alias' in result:
            print(f"  Matched alias: {result['primary_alias']}")
        print(f"  All aliases: {', '.join(result['aliases'][:5])}...")
        print()
    return results


def batch_search_mode(searcher, args):
    queries = [q.strip() for q in args.queries.split(',')]
    logger.info(f"Processing {len(queries)} queries (k={args.k})")
    batch_results = searcher.batch_search(queries, k=args.k)
    print(f"\n🔍 Batch Search Results")
    print("=" * 80)
    all_results = []
    for query, results in zip(queries, batch_results):
        print(f"\nQuery: '{query}'")
        print("-" * 40)
        if not results:
            print("  No results found.")
            continue
        for result in results[:3]:
            print(f"  {result['rank']}. {result['entity_id']}: {result['aliases'][0]}")
            print(f"     Score: {result['similarity_score']:.4f}")
        all_results.extend(results)
    return all_results


def get_entity_mode(searcher, args):
    entity = searcher.get_entity_by_id(args.entity_id)
    print(f"\n🔍 Entity Lookup: {args.entity_id}")
    print("=" * 40)
    if entity:
        print(f"Entity ID: {entity['entity_id']}")
        print(f"Aliases: {', '.join(entity['aliases'])}")
        print(f"Original Aliases: {entity['original_aliases']}")
        print(f"Processed Text: {entity['processed_text']}")
    else:
        print(f"Entity with ID '{args.entity_id}' not found in the index.")
    return entity


def similar_mode(searcher, args):
    logger.info(f"Finding entities similar to ID: {args.entity_id}")
    results = searcher.get_similar_entities(args.entity_id, k=args.k)
    print(f"\n🔍 Entities Similar to: {args.entity_id}")
    print("=" * 50)
    if not results:
        print("No similar entities found.")
        return results
    for result in results:
        print(f"Rank {result['rank']}: {result['entity_id']}")
        print(f"  Score: {result['similarity_score']:.4f}")
        print(f"  Aliases: {', '.join(result['aliases'][:3])}...")
        print()
    return results


def debug_mode(searcher, args):
    print(f"\n🔍 Debug Embedding for: '{args.query}'")
    print("=" * 50)
    stats = searcher.debug_embedding(args.query)
    print(f"Text: {stats['text']}")
    print(f"Max Length Used: {stats['max_length_used']}")
    print(f"Embedding Shape: {stats['embedding_shape']}")
    print(f"Embedding Norm: {stats['embedding_norm']:.6f}")
    print(f"Embedding Mean: {stats['embedding_mean']:.6f}")
    print(f"Embedding Std: {stats['embedding_std']:.6f}")
    print(f"Has NaN: {stats['has_nan']}")
    print(f"Has Inf: {stats['has_inf']}")
    print(f"Is Zero: {stats['is_zero']}")
    print(f"Min Value: {stats['min_value']:.6f}")
    print(f"Max Value: {stats['max_value']:.6f}")
    if stats['is_zero']:
        print("\n⚠️  WARNING: Embedding is all zeros!")
    return stats


def stats_mode(searcher, args):
    stats = searcher.get_index_stats()
    print(f"\n📊 Index Statistics")
    print("=" * 40)
    print(f"Number of Entities: {stats['num_entities']:,}")
    print(f"Embedding Dimension: {stats['embedding_dim']}")
    print(f"Index Type: {stats['index_type']}")
    print(f"Model Name: {stats['model_name']}")
    print(f"Is Trained: {stats['is_trained']}")
    print(f"Created At: {stats['created_at']}")
    print(f"Processing Time: {stats['processing_time_minutes']} minutes")
    print(f"Max Length: {stats['max_length']}")
    print(f"Use Adapter: {stats['use_adapter']}")
    print(f"Adapter Name: {stats['adapter_name']}")
    print(f"Adapter Path: {stats['adapter_path']}")
    return stats


def main():
    args = parse_arguments()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("=" * 60)
    print("SAPBERT FAISS Index Searcher (Adapter-aware)")
    print("=" * 60)

    try:
        searcher = SAPBERTIndexSearcher(
            adapter_name=args.adapter_name,
            adapter_path=args.adapter_path
        )

        logger.info(f"Loading index from: {args.index_path}")
        config = searcher.load_index(args.index_path)

        print(f"\n✅ Index loaded successfully")
        print(f"   Entities: {len(searcher.metadata):,}")
        print(f"   Model: {config.get('model_name', 'Unknown')}")
        print(f"   Type: {config.get('index_type', 'Unknown')}")
        print(f"   Max Length: {config.get('max_length', 'Unknown')}")
        print(f"   Adapter: {'ON' if searcher.use_adapter else 'OFF'}")
        if searcher.use_adapter:
            if searcher.adapter_path:
                print(f"   Adapter (Local): {searcher.adapter_path}")
            else:
                print(f"   Adapter (Hub): {searcher.adapter_name}")

        results = None
        if args.mode == 'search':
            results = search_mode(searcher, args)
        elif args.mode == 'batch_search':
            results = batch_search_mode(searcher, args)
        elif args.mode == 'get_entity':
            results = get_entity_mode(searcher, args)
        elif args.mode == 'similar':
            results = similar_mode(searcher, args)
        elif args.mode == 'stats':
            results = stats_mode(searcher, args)
        elif args.mode == 'debug':
            results = debug_mode(searcher, args)

        if args.export and results:
            searcher.export_search_results(results, args.export)
            print(f"\n💾 Results exported to: {args.export}")

    except KeyboardInterrupt:
        print(f"\n⚠️  Operation cancelled by user.")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
