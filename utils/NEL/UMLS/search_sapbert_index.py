#!/usr/bin/env python3
"""
SAPBERT Index Searcher

Search and query existing SAPBERT FAISS indexes for entity similarity.

Usage:
    python search_sapbert_index.py --index_path ./indexes/wikidata_index --query "diabetes"

Examples:
    # Single search
    python search_sapbert_index.py --query "heart disease" --k 5

    # Batch search
    python search_sapbert_index.py --mode batch_search --queries "diabetes,cancer,pneumonia"

    # Get statistics
    python search_sapbert_index.py --mode stats

    # Look up entity by QID
    python search_sapbert_index.py --mode get_entity --qid Q12206
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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SAPBERTIndexSearcher:
    """
    Utility class for searching SAPBERT FAISS indexes
    """

    def __init__(self, model_name: str = None):
        """
        Initialize the searcher

        Args:
            model_name: HuggingFace model name for SAPBERT (loaded from index config if None)
        """
        self.model_name = model_name

        # Mac-specific device selection
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                self.device = torch.device('mps')
                # Test MPS with a small tensor
                test_tensor = torch.randn(1, 1).to(self.device)
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
        self.index_config = {}

        logger.info(f"Initialized SAPBERTIndexSearcher with device: {self.device}")

        # Set multiprocessing method for Mac compatibility
        import multiprocessing as mp
        try:
            if mp.get_start_method(allow_none=True) != 'spawn':
                mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    def _load_model(self):
        """Load SAPBERT model and tokenizer if not already loaded"""
        if self.model is None or self.tokenizer is None:
            model_name = self.model_name or self.index_config.get('model_name',
                                                                  'cambridgeltl/SapBERT-from-PubMedBERT-fulltext')

            logger.info(f"Loading SAPBERT model: {model_name}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)

                # Move to device with error handling
                try:
                    self.model.to(self.device)
                except Exception as e:
                    logger.warning(f"Failed to move model to {self.device}, using CPU: {e}")
                    self.device = torch.device('cpu')
                    self.model.to(self.device)

                self.model.eval()

                # Disable gradients to save memory
                for param in self.model.parameters():
                    param.requires_grad = False

                logger.info(f"Model loaded successfully on {self.device}")

            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise

    def load_index(self, index_path: str) -> Dict:
        """
        Load FAISS index from disk

        Args:
            index_path: Path to the index files (without extension)

        Returns:
            Dictionary with index configuration
        """
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{index_path}.faiss")

            # Load metadata
            with open(f"{index_path}_metadata.pkl", 'rb') as f:
                self.metadata = pickle.load(f)

            # Load configuration
            with open(f"{index_path}_config.json", 'r') as f:
                self.index_config = json.load(f)

            self.embedding_dim = self.index_config.get('embedding_dim', 768)

            # Update model name from config if not specified
            if self.model_name is None:
                self.model_name = self.index_config.get('model_name')

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
            # Keep it simple and short for entity names
            toks = self.tokenizer.batch_encode_plus(
                [text],
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=True
            )

            # Move inputs to device safely
            try:
                toks_device = {k: v.to(self.device) for k, v in toks.items()}
            except Exception as e:
                logger.warning(f"Failed to move inputs to {self.device}, using CPU: {e}")
                self.device = torch.device('cpu')
                self.model.to(self.device)
                toks_device = {k: v.to(self.device) for k, v in toks.items()}

            with torch.no_grad():
                outputs = self.model(**toks_device)
                # Use [CLS] token embedding (first token)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

                # Clear cache if using GPU/MPS
                if self.device.type in ['cuda', 'mps']:
                    torch.cuda.empty_cache() if self.device.type == 'cuda' else None

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

        Args:
            query_text: Text to search for
            k: Number of results to return
            return_scores: Whether to include similarity scores

        Returns:
            List of search results with metadata
        """
        if self.index is None:
            raise ValueError("Index not loaded! Use load_index() first.")

        # Load model for query embedding
        self._load_model()

        # Generate query embedding
        query_embedding = self._generate_single_embedding(query_text)

        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, k)

        # Prepare results
        results = []
        seen_entities = set()  # Track entities to avoid duplicates

        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # Invalid result
                continue

            result = self.metadata[idx].copy()
            entity_id = result['entity_id']

            # Skip if we've already seen this entity (since we now have multiple entries per entity)
            if entity_id in seen_entities:
                continue
            seen_entities.add(entity_id)

            if return_scores:
                result['similarity_score'] = float(score)
            result['rank'] = len(results) + 1  # Rank based on unique entities

            # For backward compatibility, ensure 'aliases' field exists
            if 'all_aliases' in result:
                result['aliases'] = result['all_aliases']

            results.append(result)

        # Sort results by similarity score in descending order (highest first)
        if return_scores and results:
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            # Update ranks after sorting
            for i, result in enumerate(results):
                result['rank'] = i + 1

        return results

    def batch_search(self,
                     queries: List[str],
                     k: int = 10) -> List[List[Dict]]:
        """
        Search for multiple queries at once

        Args:
            queries: List of query texts
            k: Number of results per query

        Returns:
            List of search results for each query
        """
        results = []
        for query in tqdm(queries, desc="Processing queries"):
            query_results = self.search(query, k)
            results.append(query_results)
        return results

    def get_entity_by_id(self, entity_id: str) -> Optional[Dict]:
        """
        Get entity information by ID

        Args:
            entity_id: Entity ID (e.g., QID, or any other identifier)

        Returns:
            Entity metadata if found, None otherwise
        """
        # Find the first occurrence of this entity_id
        for metadata in self.metadata.values():
            if metadata['entity_id'] == entity_id:
                # Ensure we return all aliases for this entity
                if 'all_aliases' in metadata:
                    metadata['aliases'] = metadata['all_aliases']
                return metadata
        return None

    def get_index_stats(self) -> Dict:
        """Get statistics about the loaded index"""
        if self.index is None:
            return {"error": "No index loaded"}

        stats = {
            "num_entities": len(self.metadata),
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_config.get('index_type', 'Unknown'),
            "model_name": self.index_config.get('model_name', 'Unknown'),
            "is_trained": getattr(self.index, 'is_trained', True),
            "created_at": self.index_config.get('created_at', 'Unknown'),
            "processing_time_minutes": self.index_config.get('processing_time_minutes', 'Unknown')
        }

        return stats

    def get_similar_entities(self, entity_id: str, k: int = 10) -> List[Dict]:
        """
        Find entities similar to a given entity ID

        Args:
            entity_id: ID of the entity to find similar entities for
            k: Number of similar entities to return

        Returns:
            List of similar entities
        """
        entity = self.get_entity_by_id(entity_id)
        if entity is None:
            return []

        # Use the processed text of the entity as query
        query_text = entity['processed_text']
        results = self.search(query_text, k + 1)  # +1 to exclude the entity itself

        # Remove the original entity from results
        filtered_results = [r for r in results if r['entity_id'] != entity_id]
        return filtered_results[:k]

    def export_search_results(self, results: List[Dict], output_file: str = "search_results.json"):
        """
        Export search results to JSON file

        Args:
            results: Search results to export
            output_file: Output file path
        """
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Search results exported to {output_file}")

    def debug_embedding(self, text: str) -> Dict:
        """
        Debug function to check embedding generation for a specific text

        Args:
            text: Text to generate embedding for

        Returns:
            Dictionary with embedding statistics
        """
        self._load_model()

        # Generate embedding
        embedding = self._generate_single_embedding(text)

        # Calculate statistics
        stats = {
            'text': text,
            'embedding_shape': embedding.shape,
            'embedding_norm': np.linalg.norm(embedding),
            'embedding_mean': np.mean(embedding),
            'embedding_std': np.std(embedding),
            'has_nan': np.isnan(embedding).any(),
            'has_inf': np.isinf(embedding).any(),
            'is_zero': np.allclose(embedding, 0),
            'min_value': np.min(embedding),
            'max_value': np.max(embedding)
        }

        return stats


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Search SAPBERT FAISS index for entity similarity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single search with defaults
  python search_sapbert_index.py --query "diabetes"

  # Custom search
  python search_sapbert_index.py \
      --index_path ./my_indexes/biomedical_entities \
      --query "heart disease" \
      --k 5

  # Batch search
  python search_sapbert_index.py \
      --mode batch_search \
      --queries "diabetes,cancer,pneumonia,asthma" \
      --k 3

  # Get index statistics
  python search_sapbert_index.py --mode stats

  # Look up entity by QID
  python search_sapbert_index.py --mode get_entity --qid Q12206

  # Find similar entities to a QID
  python search_sapbert_index.py --mode similar --qid Q12206 --k 5

Search Modes:
  search:       Single text search (default)
  batch_search: Multiple text searches
  get_entity:   Look up entity by QID
  similar:      Find entities similar to a given QID
  stats:        Show index statistics
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['search', 'batch_search', 'get_entity', 'similar', 'stats', 'debug'],
        default='search',
        help='Search mode (default: search)'
    )

    parser.add_argument(
        '--index_path',
        type=str,
        default='./indexes/entities_index',
        help='Path to index files without extension (default: ./indexes/entities_index)'
    )

    parser.add_argument(
        '--query',
        type=str,
        default='diabetes mellitus',
        help='Search query text (default: diabetes mellitus)'
    )

    parser.add_argument(
        '--queries',
        type=str,
        default='diabetes,cancer,heart disease',
        help='Comma-separated queries for batch search (default: diabetes,cancer,heart disease)'
    )

    parser.add_argument(
        '--entity_id',
        type=str,
        default='Q12206',
        help='Entity ID for lookup or similarity search (default: Q12206)'
    )

    parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='Number of results to return (default: 10)'
    )

    parser.add_argument(
        '--export',
        type=str,
        help='Export results to JSON file (optional)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def search_mode(searcher, args):
    """Handle single search mode"""
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
    """Handle batch search mode"""
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

        for result in results[:3]:  # Show top 3 for each query
            print(f"  {result['rank']}. {result['entity_id']}: {result['aliases'][0]}")
            print(f"     Score: {result['similarity_score']:.4f}")

        all_results.extend(results)

    return all_results


def get_entity_mode(searcher, args):
    """Handle get entity mode"""
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
    """Handle similar entities mode"""
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
    """Handle debug mode"""
    print(f"\n🔍 Debug Embedding for: '{args.query}'")
    print("=" * 50)

    stats = searcher.debug_embedding(args.query)

    print(f"Text: {stats['text']}")
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
        print("\n⚠️  WARNING: Embedding is all zeros! This explains the 0.0000 similarity score.")
        print("Possible causes:")
        print("- Text too short for the model")
        print("- Tokenization issues")
        print("- Model vocabulary issues")

    return stats


def stats_mode(searcher, args):
    """Handle statistics mode"""
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

    return stats


def main():
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("=" * 60)
    print("SAPBERT FAISS Index Searcher")
    print("=" * 60)

    try:
        # Initialize searcher and load index
        logger.info(f"Loading index from: {args.index_path}")
        searcher = SAPBERTIndexSearcher()
        config = searcher.load_index(args.index_path)

        print(f"\n✅ Index loaded successfully")
        print(f"   Entities: {len(searcher.metadata):,}")
        print(f"   Model: {config.get('model_name', 'Unknown')}")
        print(f"   Type: {config.get('index_type', 'Unknown')}")

        # Execute based on mode
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

        # Export results if requested
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