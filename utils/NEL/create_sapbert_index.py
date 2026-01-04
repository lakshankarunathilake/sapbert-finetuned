#!/usr/bin/env python3
"""
SAPBERT Index Creator with Adapter Support

Creates FAISS indexes from Wikidata CSV files using SAPBERT embeddings with optional adapter support.

Usage:
    # Without adapter
    python create_sapbert_index.py --csv_path data.csv --output_dir indexes

    # With adapter
    python create_sapbert_index.py --csv_path data.csv --output_dir indexes --adapter_name my_adapter

Example:
    python create_sapbert_index.py \
        --csv_path wikidata_entities.csv \
        --output_dir ./indexes \
        --index_name biomedical_entities \
        --adapter_name domain_adapter \
        --index_type IVF \
        --batch_size 32
"""

import pandas as pd
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import pickle
import os
import json
import argparse
import time
from typing import List, Dict, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SAPBERTIndexCreator:
    """
    Utility class for creating SAPBERT embeddings and FAISS indexes
    from Wikidata CSV files with optional adapter support
    """

    def __init__(self,
                 model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                 adapter_name: Optional[str] = None,
                 adapter_path: Optional[str] = None):
        """
        Initialize the index creator

        Args:
            model_name: HuggingFace model name for SAPBERT (base model)
            adapter_name: Name of the adapter to load from HuggingFace Hub
            adapter_path: Local path to adapter weights (alternative to adapter_name)
        """
        self.model_name = model_name
        self.adapter_name = adapter_name
        self.adapter_path = adapter_path
        self.use_adapter = adapter_name is not None or adapter_path is not None

        # Mac-specific device selection to avoid MPS issues
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Use MPS on Mac M1/M2, but with caution
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
        self.embedding_dim = 768

        logger.info(f"Initialized SAPBERTIndexCreator with device: {self.device}")
        if self.use_adapter:
            logger.info(f"Adapter mode enabled: {adapter_name or adapter_path}")

        # Set multiprocessing method for Mac compatibility
        import multiprocessing as mp
        try:
            if mp.get_start_method(allow_none=True) != 'spawn':
                mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Method already set

    def _load_model(self):
        """Load SAPBERT model and tokenizer with optional adapter"""
        if self.model is None or self.tokenizer is None:
            logger.info(f"Loading SAPBERT model: {self.model_name}")

            try:
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

                # Load model with adapter support if needed
                if self.use_adapter:
                    try:
                        from adapters import AutoAdapterModel
                        logger.info("Loading model with adapters library...")

                        # Load base model with adapter support
                        self.model = AutoAdapterModel.from_pretrained(self.model_name)

                        # Load adapter
                        if self.adapter_path:
                            logger.info(f"Loading adapter from local path: {self.adapter_path}")
                            adapter_name = self.model.load_adapter(self.adapter_path, source="local")
                        else:
                            logger.info(f"Loading adapter from Hub: {self.adapter_name}")
                            adapter_name = self.model.load_adapter(self.adapter_name)

                        # Activate the adapter
                        self.model.set_active_adapters(adapter_name)
                        logger.info(f"Adapter '{adapter_name}' loaded and activated")

                        # disable the headers
                        self.model.active_head = None
                        logger.info(f"Adapter '{adapter_name}' head disabled")

                    except ImportError:
                        logger.error("adapters library not found. Install with: pip install adapters")
                        raise ImportError(
                            "The 'adapters' library is required for adapter support. "
                            "Install it with: pip install adapters"
                        )
                    except Exception as e:
                        logger.error(f"Failed to load adapter: {e}")
                        raise
                else:
                    # Load standard model without adapters
                    self.model = AutoModel.from_pretrained(self.model_name)

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

    def _preprocess_aliases(self, aliases_str: str, entity_id: str) -> List[str]:
        """
        Preprocess aliases by filtering and cleaning

        Args:
            aliases_str: String of aliases joined with ||
            entity_id: Entity ID to remove from aliases

        Returns:
            List of cleaned aliases
        """
        if pd.isna(aliases_str) or not aliases_str.strip():
            return []

        # Split by ||
        aliases = aliases_str.split('||')

        # Clean and filter aliases
        cleaned_aliases = []
        for alias in aliases:
            alias = alias.strip()
            if alias and alias != entity_id:  # Remove empty aliases and ID matches
                cleaned_aliases.append(alias)

        return cleaned_aliases

    def _generate_embeddings_batch(self, texts: List[str], batch_size: int = 16, max_length: int = 16) -> np.ndarray:
        """
        Generate embeddings for multiple texts in batches using efficient tokenization

        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            max_length: Maximum sequence length for tokenization (shorter is better for entity names)

        Returns:
            Array of embeddings
        """
        self._load_model()
        embeddings = []

        # Reduce batch size on Mac to avoid memory issues
        # TODO: Activate this if needed
        # import platform
        # if platform.system() == "Darwin":  # macOS
        #     batch_size = min(batch_size, 8)
        #     logger.info(f"Running on macOS, reducing batch size to {batch_size}")

        logger.info(f"Using max_length={max_length} for tokenization")

        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]

            try:
                # Use simple, efficient tokenization
                toks = self.tokenizer.batch_encode_plus(
                    batch_texts,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=True
                )

                # Log tokenization info for debugging (first batch only)
                if i == 0:
                    logger.info(f"Tokenization - Sequence length: {toks['input_ids'].shape[1]}")
                    logger.info(f"Sample text: '{batch_texts[0]}'")
                    logger.info(f"Sample tokens: {toks['input_ids'][0].tolist()}")

                # Move inputs to device safely
                try:
                    toks_device = {k: v.to(self.device) for k, v in toks.items()}
                except Exception as e:
                    logger.warning(f"Failed to move inputs to {self.device}, using CPU: {e}")
                    self.device = torch.device('cpu')
                    self.model.to(self.device)
                    toks_device = {k: v.to(self.device) for k, v in toks.items()}

                # Generate embeddings with memory management
                with torch.no_grad():
                    try:
                        outputs = self.model(**toks_device)
                        # Use [CLS] token embedding (first token)
                        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

                        # Check for problematic embeddings in first few batches
                        if i < 3:
                            for j, emb in enumerate(batch_embeddings):
                                norm = np.linalg.norm(emb)
                                if norm < 1e-6:
                                    logger.warning(
                                        f"Very small embedding norm ({norm:.8f}) for text: '{batch_texts[j]}'")
                                elif np.isnan(emb).any():
                                    logger.warning(f"NaN in embedding for text: '{batch_texts[j]}'")

                        embeddings.extend(batch_embeddings)

                        # Clear GPU cache if using CUDA/MPS
                        if self.device.type in ['cuda', 'mps']:
                            torch.cuda.empty_cache() if self.device.type == 'cuda' else None

                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            logger.warning(f"Out of memory, processing individually")
                            # Process one by one as fallback
                            for text in batch_texts:
                                single_toks = self.tokenizer.batch_encode_plus(
                                    [text],
                                    padding="max_length",
                                    max_length=max_length,
                                    truncation=True,
                                    return_tensors="pt",
                                    add_special_tokens=True
                                )
                                single_toks_device = {k: v.to(self.device) for k, v in single_toks.items()}

                                with torch.no_grad():
                                    single_output = self.model(**single_toks_device)
                                    single_embedding = single_output.last_hidden_state[:, 0, :].cpu().numpy()
                                    embeddings.extend(single_embedding)
                        else:
                            raise e

            except Exception as e:
                logger.error(f"Error processing batch {i // batch_size + 1}: {e}")
                raise

        logger.info(f"Generated {len(embeddings)} embeddings")
        return np.array(embeddings)

    def _build_faiss_index(self, embeddings: np.ndarray, index_type: str):
        """Build FAISS index based on the specified type"""
        n_embeddings = embeddings.shape[0]

        logger.info(f"Building {index_type} index with {n_embeddings:,} embeddings...")
        logger.info(f"Embedding dimension: {self.embedding_dim}")

        # Normalize embeddings for cosine similarity
        logger.info("Normalizing embeddings for cosine similarity...")
        faiss.normalize_L2(embeddings)

        if index_type == "Flat":
            logger.info("Creating Flat index (exact search)...")
            index = faiss.IndexFlatIP(self.embedding_dim)

        elif index_type == "IVF":
            nlist = min(int(np.sqrt(n_embeddings)), 1000)
            logger.info(f"Creating IVF index with {nlist} clusters...")
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)

            logger.info("Training IVF index (this may take a few minutes)...")
            start_time = time.time()
            index.train(embeddings.astype(np.float32))
            training_time = time.time() - start_time
            logger.info(f"IVF training completed in {training_time:.1f} seconds")

        elif index_type == "HNSW":
            M = 32
            logger.info(f"Creating HNSW index with M={M} connections...")
            logger.info("Note: HNSW construction can be slow but provides very fast search")
            index = faiss.IndexHNSWFlat(self.embedding_dim, M)
            index.hnsw.efConstruction = 200
            logger.info(f"HNSW parameters: M={M}, efConstruction={index.hnsw.efConstruction}")

        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        # Add embeddings to index with progress tracking
        logger.info("Adding embeddings to index...")
        start_time = time.time()

        if index_type == "HNSW" and n_embeddings > 1000:
            chunk_size = max(100, n_embeddings // 20)
            logger.info(f"Adding {n_embeddings:,} embeddings in chunks of {chunk_size:,}")

            for i in tqdm(range(0, n_embeddings, chunk_size), desc="Building HNSW index"):
                end_idx = min(i + chunk_size, n_embeddings)
                chunk = embeddings[i:end_idx].astype(np.float32)
                index.add(chunk)

                if (i // chunk_size) % 5 == 0:
                    elapsed = time.time() - start_time
                    progress = (i + chunk_size) / n_embeddings
                    estimated_total = elapsed / progress if progress > 0 else 0
                    remaining = estimated_total - elapsed
                    logger.info(f"Progress: {progress * 100:.1f}% - ETA: {remaining / 60:.1f} minutes")
        else:
            if n_embeddings > 10000:
                logger.info(f"Adding {n_embeddings:,} embeddings (this may take several minutes)...")
            index.add(embeddings.astype(np.float32))

        construction_time = time.time() - start_time
        logger.info(
            f"Index construction completed in {construction_time:.1f} seconds ({construction_time / 60:.1f} minutes)")
        logger.info(f"Built {index_type} index with {index.ntotal:,} embeddings")

        if hasattr(index, 'is_trained'):
            logger.info(f"Index is_trained: {index.is_trained}")

        return index

    def _save_index(self, index, metadata: Dict, index_path: str, config: Dict):
        """Save FAISS index, metadata, and configuration to disk"""
        # Save FAISS index
        faiss.write_index(index, f"{index_path}.faiss")

        # Save metadata
        with open(f"{index_path}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)

        # Save configuration
        with open(f"{index_path}_config.json", 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Index saved to {index_path}")

    def create_index_from_csv(self,
                              csv_path: str,
                              output_dir: str,
                              index_name: str = "entities_index",
                              index_type: str = "IVF",
                              batch_size: int = 16,
                              max_length: int = 16) -> str:
        """
        Create FAISS index from CSV file and save to disk

        Args:
            csv_path: Path to the CSV file (first column = ID, second column = aliases)
            output_dir: Directory to save the index and metadata
            index_name: Name for the index files
            index_type: Type of FAISS index ("Flat", "IVF", "HNSW")
            batch_size: Batch size for embedding generation
            max_length: Maximum sequence length for tokenization

        Returns:
            Path to the saved index directory
        """
        start_time = time.time()

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load and validate CSV
        logger.info(f"Loading CSV from {csv_path}")
        df = pd.read_csv(csv_path)

        if len(df.columns) < 2:
            raise ValueError(f"CSV must have at least 2 columns. Found {len(df.columns)} columns: {list(df.columns)}")

        # Use first two columns regardless of their names
        id_column = df.columns[0]
        aliases_column = df.columns[1]

        logger.info(f"Using columns: ID='{id_column}', Aliases='{aliases_column}'")
        logger.info(f"Loaded {len(df)} rows")

        # Process data
        processed_data = []
        all_texts = []

        logger.info("Processing aliases...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            entity_id = str(row[id_column]).strip()
            aliases_str = str(row[aliases_column]) if not pd.isna(row[aliases_column]) else ""

            if not entity_id or entity_id.lower() in ['nan', 'none', '']:
                continue

            aliases = self._preprocess_aliases(aliases_str, entity_id)

            if not aliases:
                continue

            # Create separate index entries for each alias
            for alias in aliases:
                all_texts.append(alias)

                processed_data.append({
                    'entity_id': entity_id,
                    'primary_alias': alias,
                    'all_aliases': aliases,
                    'original_aliases': aliases_str,
                    'processed_text': alias,
                    'index_id': len(processed_data)
                })

        logger.info(
            f"Processed {len(processed_data)} alias entries from {len(set([d['entity_id'] for d in processed_data]))} unique entities")

        if len(processed_data) == 0:
            raise ValueError("No valid entries found after processing. Check your CSV format and data.")

        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self._generate_embeddings_batch(all_texts, batch_size, max_length)
        self.embedding_dim = embeddings.shape[1]

        # Build FAISS index
        logger.info(f"Building {index_type} index...")
        index = self._build_faiss_index(embeddings, index_type)

        # Prepare metadata
        metadata = {i: processed_data[i] for i in range(len(processed_data))}

        # Save to disk
        index_path = os.path.join(output_dir, index_name)
        config = {
            'index_type': index_type,
            'model_name': self.model_name,
            'adapter_name': self.adapter_name,
            'adapter_path': self.adapter_path,
            'use_adapter': self.use_adapter,
            'embedding_dim': self.embedding_dim,
            'num_entities': len(processed_data),
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'processing_time_minutes': (time.time() - start_time) / 60,
            'max_length': max_length,
            'source_columns': {
                'id_column': id_column,
                'aliases_column': aliases_column
            }
        }

        self._save_index(index, metadata, index_path, config)

        # Save processed data for reference
        processed_df = pd.DataFrame(processed_data)
        processed_df.to_csv(os.path.join(output_dir, f"{index_name}_processed_data.csv"), index=False)

        end_time = time.time()
        processing_time = (end_time - start_time) / 60

        logger.info(f"Index creation completed in {processing_time:.1f} minutes")
        logger.info(f"Index saved to: {index_path}")

        return index_path


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Create SAPBERT FAISS index from Wikidata CSV file with optional adapter support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage without adapter
  python create_sapbert_index.py --csv_path data.csv

  # With adapter from HuggingFace Hub
  python create_sapbert_index.py \
      --csv_path data.csv \
      --adapter_name username/my-adapter

  # With local adapter
  python create_sapbert_index.py \
      --csv_path data.csv \
      --adapter_path ./my_adapter

  # Custom parameters with adapter
  python create_sapbert_index.py \
      --csv_path my_data.csv \
      --output_dir ./my_indexes \
      --index_name biomedical_entities \
      --adapter_name domain_adapter \
      --index_type IVF \
      --batch_size 32

Index Types (Accuracy vs Speed):
  Flat: Maximum accuracy (100% exact), slower search
  IVF:  High accuracy (~95-99%), good search speed - RECOMMENDED
  HNSW: Good accuracy (~90-95%), fastest search
        """
    )

    parser.add_argument(
        '--csv_path',
        type=str,
        default='data/entities.csv',
        help='Path to CSV file (first column = ID, second column = aliases with ||)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./indexes',
        help='Directory to save index files'
    )

    parser.add_argument(
        '--index_name',
        type=str,
        default='entities_index',
        help='Name for index files'
    )

    parser.add_argument(
        '--index_type',
        type=str,
        choices=['Flat', 'IVF', 'HNSW'],
        default='IVF',
        help='FAISS index type'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for embedding generation'
    )

    parser.add_argument(
        '--model_name',
        type=str,
        default='cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
        help='SAPBERT base model name'
    )

    parser.add_argument(
        '--adapter_name',
        type=str,
        default=None,
        help='Adapter name from HuggingFace Hub (e.g., username/adapter-name)'
    )

    parser.add_argument(
        '--adapter_path',
        type=str,
        default=None,
        help='Local path to adapter weights (alternative to --adapter_name)'
    )

    parser.add_argument(
        '--validate_only',
        action='store_true',
        help='Only validate CSV file without creating index'
    )

    parser.add_argument(
        '--max_length',
        type=int,
        default=16,
        help='Maximum sequence length for tokenization'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def validate_csv_file(csv_path: str):
    """Validate CSV file format and content"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    try:
        df_sample = pd.read_csv(csv_path, nrows=5)

        if len(df_sample.columns) < 2:
            raise ValueError(
                f"CSV must have at least 2 columns. Found {len(df_sample.columns)} columns: {list(df_sample.columns)}")

        id_column = df_sample.columns[0]
        aliases_column = df_sample.columns[1]

        row_count = len(pd.read_csv(csv_path))

        logger.info(f"✓ CSV file validated: {csv_path}")
        logger.info(f"✓ Found {len(df_sample.columns)} columns: {list(df_sample.columns)}")
        logger.info(f"✓ Using: ID column='{id_column}', Aliases column='{aliases_column}'")
        logger.info(f"✓ Total rows: {row_count:,}")
        logger.info(f"✓ Sample data preview:")

        for i, (_, row) in enumerate(df_sample.iterrows()):
            if i >= 3:
                break
            entity_id = str(row[id_column]).strip()
            aliases = str(row[aliases_column]) if not pd.isna(row[aliases_column]) else ""
            logger.info(f"    {entity_id}: {aliases[:100]}{'...' if len(aliases) > 100 else ''}")

        return row_count

    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")


def main():
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate adapter arguments
    if args.adapter_name and args.adapter_path:
        logger.error("Cannot specify both --adapter_name and --adapter_path. Choose one.")
        return

    # Mac-specific environment setup
    import platform
    if platform.system() == "Darwin":
        os.environ["OMP_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        logger.info("Applied macOS-specific optimizations")

    print("=" * 60)
    print("SAPBERT FAISS Index Creator with Adapter Support")
    print("=" * 60)

    try:
        # Validate CSV
        logger.info(f"Validating CSV file: {args.csv_path}")
        row_count = validate_csv_file(args.csv_path)

        if args.validate_only:
            print("✅ CSV validation completed successfully!")
            return

        # Estimate processing time
        estimated_minutes = max(1, row_count // 500)

        # Display configuration
        print(f"\n⚙️  Configuration:")
        print(f"   CSV File: {args.csv_path}")
        print(f"   Output Directory: {args.output_dir}")
        print(f"   Index Name: {args.index_name}")
        print(f"   Index Type: {args.index_type}")
        print(f"   Base Model: {args.model_name}")

        if args.adapter_name:
            print(f"   Adapter (Hub): {args.adapter_name}")
        elif args.adapter_path:
            print(f"   Adapter (Local): {args.adapter_path}")
        else:
            print(f"   Adapter: None (using base model only)")

        print(f"   Batch Size: {args.batch_size}")
        print(f"   Rows to Process: {row_count:,}")
        print(f"   Estimated Time: ~{estimated_minutes} minutes")

        if platform.system() == "Darwin":
            print(f"   Platform: macOS (optimized settings applied)")

        # Confirm for large datasets
        if row_count > 10000:
            response = input(f"\n⚠️  Large dataset detected ({row_count:,} rows). Continue? [y/N]: ")
            if response.lower() != 'y':
                print("Operation cancelled.")
                return

        # Create index
        print(f"\n🚀 Creating index...")
        creator = SAPBERTIndexCreator(
            model_name=args.model_name,
            adapter_name=args.adapter_name,
            adapter_path=args.adapter_path
        )

        index_path = creator.create_index_from_csv(
            csv_path=args.csv_path,
            output_dir=args.output_dir,
            index_name=args.index_name,
            index_type=args.index_type,
            batch_size=args.batch_size,
            max_length=args.max_length
        )

        # Show success message
        print(f"\n✅ Index creation completed!")
        print(f"   Index saved to: {index_path}")

        # List created files
        from pathlib import Path
        index_files = list(Path(args.output_dir).glob(f"{args.index_name}*"))
        print(f"\n📁 Created Files:")
        for file_path in sorted(index_files):
            file_size = file_path.stat().st_size / (1024 * 1024)
            print(f"   {file_path.name} ({file_size:.1f} MB)")

        # Show usage example
        print(f"\n🔍 Next Steps:")
        print(f"Use the search utility to query your index:")
        print(f"python search_sapbert_index.py --index_path {index_path} --query \"your search term\"")

    except KeyboardInterrupt:
        print(f"\n⚠️  Operation cancelled by user.")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()