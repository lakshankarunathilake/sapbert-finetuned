#!/usr/bin/env python3
"""
SAPBERT Evaluation Script for BC5CDR Dataset (Chemical and Disease)

Now adapter-aware and aligned with utils.py evaluation methodology:
- Supports top-k candidate evaluation
- Calculates acc@1, acc@2, ..., acc@k metrics
- Uses utils.py data structure and evaluation logic
- Optional CLI overrides: --adapter_name or --adapter_path (mutually exclusive)

Usage:
    python eval_sapbert_bc5cdr.py --index_path ./utils/NEL/UMLS/indexes/Wikidata --data_dir ./data/bc5cdr-chemical
    python eval_sapbert_bc5cdr.py --index_path ./utils/NEL/UMLS/indexes/Wikidata --data_dir ./data/bc5cdr-disease --topk 10
    # Override adapter from Hub
    python eval_sapbert_bc5cdr.py --index_path ... --data_dir ... --adapter_name username/my-adapter
    # Override adapter from local path
    python eval_sapbert_bc5cdr.py --index_path ... --data_dir ... --adapter_path ./my_adapter
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import numpy as np
from collections import defaultdict, Counter
import pandas as pd


# Fix OpenMP library conflict on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add the project root directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.NEL.search_sapbert_index import SAPBERTIndexSearcher


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_label(predicted_cui: str, golden_cui: str) -> int:
    """
    Check if predicted CUI matches golden CUI (from utils.py)

    Some composite annotation didn't consider orders
    So, set label '1' if any cui is matched within composite cui (or single cui)
    Otherwise, set label '0'
    """
    return int(len(set(predicted_cui.split("|")).intersection(set(golden_cui.split("|")))) > 0)


def check_k(queries: List[Dict]) -> int:
    """
    Get the number of candidates from the first query (from utils.py)
    """
    return len(queries[0]['mentions'][0]['candidates'])


def evaluate_topk_acc(data: Dict) -> Dict:
    """
    Evaluate acc@1~acc@k (from utils.py)
    """
    queries = data['queries']
    k = check_k(queries)

    for i in range(0, k):
        hit = 0
        for query in queries:
            mentions = query['mentions']
            mention_hit = 0
            for mention in mentions:
                candidates = mention['candidates'][:i+1]  # to get acc@(i+1)
                mention_hit += np.any([candidate['label'] for candidate in candidates])

            # When all mentions in a query are predicted correctly,
            # we consider it as a hit
            if mention_hit == len(mentions):
                hit += 1

        data['acc{}'.format(i+1)] = hit / len(queries)

    return data


class BC5CDREvaluator:
    """
    Evaluator for BC5CDR dataset (Chemical and Disease) using SAPBERT search functionality
    """

    def __init__(self, index_path: str, data_dir: str,
                 adapter_name: Optional[str] = None,
                 adapter_path: Optional[str] = None):
        """
        Initialize the evaluator

        Args:
            index_path: Path to the SAPBERT index files (without extension)
            data_dir: Path to the BC5CDR data directory
            adapter_name: Optional HF Hub adapter name (override)
            adapter_path: Optional local adapter path (override)
        """
        self.index_path = index_path
        self.data_dir = data_dir
        self.dataset_name = os.path.basename(data_dir)
        self.searcher: Optional[SAPBERTIndexSearcher] = None
        self.adapter_name = adapter_name
        self.adapter_path = adapter_path

        self.test_queries = []
        self.results = {
            'total_queries': 0,
            'correct_predictions': 0,
            'top1_accuracy': 0.0,
            'error_cases': [],
            'error_analysis': {},
            'performance_metrics': {},
            # For reproducibility
            'runtime_config': {}
        }

    def load_test_queries(self) -> List[Dict]:
        """
        Load test queries from processed test files
        
        NOW CREATES ONE QUERY PER INDIVIDUAL MENTION OCCURRENCE
        (matching count_bc5cdr_mentions.py methodology)

        Returns:
            List of test query dictionaries
        """
        processed_test_dir = os.path.join(self.data_dir, 'processed_test')
        logger.info(f"Loading test queries from: {processed_test_dir}")

        queries = []
        concept_files = [f for f in os.listdir(processed_test_dir) if f.endswith('.concept')]
        logger.info(f"Found {len(concept_files)} concept files to process")

        # Track statistics like count_bc5cdr_mentions.py
        total_raw_mentions = 0
        all_mention_texts = set()
        all_cuis = set()
        mention_counts = Counter()
        cui_counts = Counter()

        for concept_file in tqdm(concept_files, desc="Loading test queries"):
            file_path = os.path.join(processed_test_dir, concept_file)

            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('||')
                if len(parts) == 5:  # doc_id||start_pos|end_pos||entity_type||mention_text||cui
                    doc_id = parts[0]
                    position = parts[1]  # start_pos|end_pos
                    entity_type = parts[2]
                    mention_text = parts[3]  # Index 3 is the entity name
                    cui = parts[4]  # Index 4 is the matching CUI

                    # Count raw mentions (like count_bc5cdr_mentions.py)
                    total_raw_mentions += 1
                    mention_text_normalized = mention_text.lower()
                    all_mention_texts.add(mention_text_normalized)
                    all_cuis.add(cui)
                    mention_counts[mention_text_normalized] += 1
                    cui_counts[cui] += 1

                    # Create ONE QUERY per individual mention occurrence
                    # (This is the key change - no deduplication!)
                    queries.append({
                        'doc_id': doc_id,
                        'position': position,
                        'entity_type': entity_type,
                        'query_text': mention_text,
                        'golden_cui': cui,
                        'concept_file': concept_file
                    })

        # Log detailed statistics
        logger.info("="*60)
        logger.info("DATASET STATISTICS (count_bc5cdr_mentions.py methodology)")
        logger.info("="*60)
        logger.info(f"Total raw mentions in files: {total_raw_mentions:,}")
        logger.info(f"Unique mentions (case-insensitive): {len(all_mention_texts):,}")
        logger.info(f"Unique CUIs: {len(all_cuis):,}")
        logger.info(f"Average mentions per CUI: {total_raw_mentions / len(all_cuis):.2f}" if len(all_cuis) > 0 else "N/A")
        logger.info("")
        logger.info("EVALUATION QUERY STATISTICS (NO deduplication)")
        logger.info("="*60)
        logger.info(f"Total evaluation queries: {len(queries):,}")
        logger.info(f"Queries == Raw mentions: {len(queries) == total_raw_mentions}")
        logger.info("")
        logger.info("EVALUATION APPROACH:")
        logger.info("- Each individual mention occurrence is evaluated separately")
        logger.info("- No deduplication by (document, CUI) pairs")
        logger.info("- Matches count_bc5cdr_mentions.py counting methodology")
        logger.info("="*60)
        logger.info("")

        # Store statistics in results
        self.results['dataset_statistics'] = {
            'total_raw_mentions': total_raw_mentions,
            'unique_mentions': len(all_mention_texts),
            'unique_cuis': len(all_cuis),
            'total_queries': len(queries),
            'queries_equal_mentions': len(queries) == total_raw_mentions,
            'top_10_mentions': [
                {'mention': mention, 'count': count} 
                for mention, count in mention_counts.most_common(10)
            ],
            'top_10_cuis': [
                {'cui': cui, 'count': count}
                for cui, count in cui_counts.most_common(10)
            ]
        }

        logger.info(f"Loaded {len(queries)} test queries (one per mention occurrence)")
        return queries

    def initialize_searcher(self):
        """Initialize the SAPBERT searcher (adapter-aware)"""
        logger.info(f"Initializing SAPBERT searcher with index: {self.index_path}")

        # Pass adapter overrides to searcher (will fall back to index config if None)
        self.searcher = SAPBERTIndexSearcher(
            adapter_name=self.adapter_name,
            adapter_path=self.adapter_path
        )
        config = self.searcher.load_index(self.index_path)

        # Record runtime config for reproducibility
        runtime_cfg = {
            'model_name': config.get('model_name', 'Unknown'),
            'index_type': config.get('index_type', 'Unknown'),
            'max_length': config.get('max_length', 'Unknown'),
            'use_adapter_effective': self.searcher.use_adapter,
            'adapter_name_effective': self.searcher.adapter_name,
            'adapter_path_effective': self.searcher.adapter_path,
            'use_adapter_from_index_config': bool(config.get('use_adapter', False)),
            'adapter_name_from_index_config': config.get('adapter_name'),
            'adapter_path_from_index_config': config.get('adapter_path'),
        }
        self.results['runtime_config'] = runtime_cfg

        logger.info(f"Index loaded. Model: {runtime_cfg['model_name']}. "
                    f"Adapter={'ON' if runtime_cfg['use_adapter_effective'] else 'OFF'} "
                    f"({runtime_cfg['adapter_path_effective'] or runtime_cfg['adapter_name_effective']})")

    def predict_topk(self, eval_queries: List[Dict], topk: int = 10) -> Dict:
        """
        Predict top-k candidates for each query (aligned with utils.py)

        Args:
            eval_queries: List of query dictionaries
            topk: Number of top candidates to retrieve

        Returns:
            Dictionary with queries and their candidates (utils.py format)
        """
        logger.info(f"Retrieving top-{topk} candidates for each query")

        queries = []
        for eval_query in tqdm(eval_queries, desc="Evaluating queries"):
            query_text = eval_query['query_text']
            golden_cui = eval_query['golden_cui']

            # Search for top-k results
            search_results = self.searcher.search(query_text, k=topk, return_scores=True)

            # Build candidates list
            dict_candidates = []
            for result in search_results:
                predicted_cui = result.get('entity_id', '')
                predicted_name = result.get('aliases', [''])[0] if result.get('aliases') else ''
                similarity_score = result.get('similarity_score', 0.0)

                # Use check_label function from utils.py
                label = check_label(predicted_cui, golden_cui)

                dict_candidates.append({
                    'name': predicted_name,
                    'labelcui': predicted_cui,
                    'label': label,
                    'similarity_score': similarity_score
                })

            # Handle case where no results found - create empty candidates
            if not dict_candidates:
                dict_candidates = [{
                    'name': '',
                    'labelcui': '',
                    'label': 0,
                    'similarity_score': 0.0
                } for _ in range(topk)]

            # Build mention structure (utils.py format)
            dict_mentions = [{
                'mention': query_text,
                'golden_cui': golden_cui,
                'candidates': dict_candidates
            }]

            # Build query structure (utils.py format)
            queries.append({
                'mentions': dict_mentions,
                'metadata': {
                    'doc_id': eval_query.get('doc_id'),
                    'position': eval_query.get('position'),
                    'entity_type': eval_query.get('entity_type'),
                    'concept_file': eval_query.get('concept_file')
                }
            })

        result = {
            'queries': queries
        }

        return result

    def run_evaluation(self, topk: int = 10) -> Dict:
        """
        Run the complete evaluation (aligned with utils.py)

        Args:
            topk: Number of top candidates to retrieve for acc@k evaluation
        """
        logger.info(f"Starting BC5CDR-{self.dataset_name} evaluation")

        # Load data
        self.test_queries = self.load_test_queries()

        # Initialize searcher
        self.initialize_searcher()

        # Run prediction (utils.py style)
        logger.info(f"Evaluating {len(self.test_queries)} queries with top-{topk} candidates")
        result = self.predict_topk(self.test_queries, topk=topk)

        # Calculate acc@k metrics (utils.py style)
        result = evaluate_topk_acc(result)

        # Store results
        self.results.update({
            'queries': result['queries'],
            'total_queries': len(result['queries'])
        })

        # Add acc@k metrics
        for i in range(1, topk + 1):
            acc_key = f'acc{i}'
            if acc_key in result:
                self.results[acc_key] = result[acc_key]

        # Calculate additional statistics
        total_queries = len(result['queries'])
        correct_at_1 = int(result.get('acc1', 0) * total_queries)

        # Analyze errors (only for top-1)
        error_cases = []
        error_counts = Counter()
        similarity_scores = []

        for query_result in result['queries']:
            mention = query_result['mentions'][0]
            top_candidate = mention['candidates'][0] if mention['candidates'] else None

            if top_candidate:
                similarity_scores.append(top_candidate.get('similarity_score', 0.0))

                if top_candidate['label'] == 0:
                    # Error case
                    error_type = self.categorize_error(
                        top_candidate['labelcui'],
                        mention['golden_cui'],
                        mention['mention'],
                        top_candidate['name']
                    )
                    error_counts[error_type] += 1
                    error_cases.append({
                        'query': query_result.get('metadata', {}),
                        'mention': mention['mention'],
                        'golden_cui': mention['golden_cui'],
                        'predicted_cui': top_candidate['labelcui'],
                        'predicted_name': top_candidate['name'],
                        'similarity_score': top_candidate.get('similarity_score', 0.0),
                        'error_type': error_type
                    })

        self.results.update({
            'correct_predictions': correct_at_1,
            'top1_accuracy': result.get('acc1', 0.0),
            'error_cases': error_cases,
            'error_analysis': {
                'error_counts': dict(error_counts),
                'error_rate': 1 - result.get('acc1', 0.0),
                'avg_similarity_score': np.mean(similarity_scores) if similarity_scores else 0.0,
                'min_similarity_score': np.min(similarity_scores) if similarity_scores else 0.0,
                'max_similarity_score': np.max(similarity_scores) if similarity_scores else 0.0
            },
            'performance_metrics': {
                'total_queries': total_queries,
                'correct_predictions': correct_at_1,
                'incorrect_predictions': total_queries - correct_at_1,
                'top1_accuracy': result.get('acc1', 0.0),
                'error_rate': 1 - result.get('acc1', 0.0)
            }
        })

        # Log all acc@k metrics
        logger.info(f"Evaluation completed:")
        for i in range(1, min(topk + 1, 11)):  # Log up to acc@10
            acc_key = f'acc{i}'
            if acc_key in result:
                logger.info(f"  acc@{i}: {result[acc_key]:.4f}")

        return self.results

    def categorize_error(self, predicted_cui: str, golden_cui: str, query_text: str, predicted_name: str) -> str:
        """
        Categorize the type of error
        """
        if not predicted_cui:
            return 'no_prediction'
        if predicted_name and query_text.lower() in predicted_name.lower():
            return 'partial_match'
        if predicted_cui != golden_cui:
            return 'wrong_entity'
        return 'unknown_error'

    def generate_error_report(self, output_dir: str = None) -> str:
        """
        Generate detailed error analysis report
        """
        if output_dir is None:
            output_dir = os.path.dirname(__file__)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        error_data = []
        for error_case in self.results['error_cases']:
            query_metadata = error_case.get('query', {})
            error_data.append({
                'doc_id': query_metadata.get('doc_id', 'N/A'),
                'query_text': error_case.get('mention', 'N/A'),
                'golden_cui': error_case.get('golden_cui', 'N/A'),
                'predicted_cui': error_case.get('predicted_cui', 'N/A'),
                'predicted_name': error_case.get('predicted_name', 'N/A'),
                'similarity_score': error_case.get('similarity_score', 0.0),
                'error_type': error_case.get('error_type', 'unknown'),
                'concept_file': query_metadata.get('concept_file', 'N/A')
            })

        error_df = pd.DataFrame(error_data)

        report_path = os.path.join(output_dir, f'{self.dataset_name}_error_analysis.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"BC5CDR-{self.dataset_name.title()} SAPBERT Evaluation Error Analysis\n")
            f.write("=" * 60 + "\n\n")

            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Queries: {self.results['total_queries']}\n")
            f.write(f"Correct Predictions: {self.results['correct_predictions']}\n")
            f.write(f"Incorrect Predictions: {len(self.results['error_cases'])}\n")
            f.write(f"Top-1 Accuracy: {self.results['top1_accuracy']:.4f}\n")
            f.write(f"Error Rate: {self.results['error_analysis']['error_rate']:.4f}\n\n")

            # Adapter/runtime config
            rcfg = self.results.get('runtime_config', {})
            f.write("RUNTIME CONFIGURATION\n")
            f.write("-" * 25 + "\n")
            f.write(f"Model Name: {rcfg.get('model_name')}\n")
            f.write(f"Index Type: {rcfg.get('index_type')}\n")
            f.write(f"Max Length: {rcfg.get('max_length')}\n")
            f.write(f"Adapter Enabled (effective): {rcfg.get('use_adapter_effective')}\n")
            f.write(f"Adapter Name (effective): {rcfg.get('adapter_name_effective')}\n")
            f.write(f"Adapter Path (effective): {rcfg.get('adapter_path_effective')}\n")
            f.write(f"Adapter (from index config): {rcfg.get('use_adapter_from_index_config')}\n")
            f.write(f"Adapter Name (index): {rcfg.get('adapter_name_from_index_config')}\n")
            f.write(f"Adapter Path (index): {rcfg.get('adapter_path_from_index_config')}\n\n")

            # Acc@k metrics
            f.write("ACCURACY@K METRICS\n")
            f.write("-" * 25 + "\n")
            for i in range(1, 11):
                acc_key = f'acc{i}'
                if acc_key in self.results:
                    f.write(f"acc@{i}: {self.results[acc_key]:.4f}\n")
            f.write("\n")

            # Error type distribution
            f.write("ERROR TYPE DISTRIBUTION\n")
            f.write("-" * 25 + "\n")
            for error_type, count in self.results['error_analysis']['error_counts'].items():
                percentage = (count / len(self.results['error_cases'])) * 100 if self.results['error_cases'] else 0
                f.write(f"{error_type}: {count} ({percentage:.1f}%)\n")
            f.write("\n")

            # Similarity score statistics
            f.write("SIMILARITY SCORE STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average: {self.results['error_analysis']['avg_similarity_score']:.4f}\n")
            f.write(f"Minimum: {self.results['error_analysis']['min_similarity_score']:.4f}\n")
            f.write(f"Maximum: {self.results['error_analysis']['max_similarity_score']:.4f}\n\n")

            # Detailed error cases
            f.write("DETAILED ERROR CASES\n")
            f.write("-" * 20 + "\n")
            for i, error_case in enumerate(self.results['error_cases'][:50], 1):
                query_metadata = error_case.get('query', {})
                f.write(f"\nError Case {i}:\n")
                f.write(f"  Document ID: {query_metadata.get('doc_id', 'N/A')}\n")
                f.write(f"  Query Text: '{error_case.get('mention', 'N/A')}'\n")
                f.write(f"  Golden CUI: {error_case.get('golden_cui', 'N/A')}\n")
                f.write(f"  Predicted CUI: {error_case.get('predicted_cui', 'N/A')}\n")
                f.write(f"  Predicted Name: '{error_case.get('predicted_name', 'N/A')}'\n")
                f.write(f"  Similarity Score: {error_case.get('similarity_score', 0.0):.4f}\n")
                f.write(f"  Error Type: {error_case.get('error_type', 'unknown')}\n")
                f.write(f"  Concept File: {query_metadata.get('concept_file', 'N/A')}\n")

            if len(self.results['error_cases']) > 50:
                f.write(f"\n... and {len(self.results['error_cases']) - 50} more error cases\n")

        logger.info(f"Error analysis report saved to: {report_path}")
        return report_path

    def save_results(self, output_dir: str = None) -> str:
        """
        Save evaluation results to JSON file
        """
        if output_dir is None:
            output_dir = os.path.dirname(__file__)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        results_path = os.path.join(output_dir, f'{self.dataset_name}_evaluation_results.json')

        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        serializable_results = json.loads(json.dumps(self.results, default=convert_numpy))

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Evaluation results saved to: {results_path}")
        return results_path


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate SAPBERT on BC5CDR dataset (Chemical and Disease) [adapter-aware]',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Chemical dataset
  python eval_sapbert_bc5cdr_fixed.py --index_path ./utils/NEL/UMLS/indexes/Wikidata --data_dir ./data/bc5cdr-chemical

  # With adapter override (Hub)
  python eval_sapbert_bc5cdr_fixed.py --index_path ... --data_dir ... --adapter_name username/my-adapter

  # With adapter override (local)
  python eval_sapbert_bc5cdr_fixed.py --index_path ... --data_dir ... --adapter_path ./my_adapter
        """
    )

    parser.add_argument('--index_path', type=str, required=True,
                        help='Path to SAPBERT index files (without extension)')

    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to BC5CDR data directory (chemical or disease)')

    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results and reports (default: evaluation directory)')

    parser.add_argument('--topk', type=int, default=10,
                        help='Number of top-k candidates to retrieve (default: 10)')

    # NEW: adapter overrides (mutually exclusive)
    parser.add_argument('--adapter_name', type=str, default=None,
                        help='Adapter name from HuggingFace Hub (overrides index config)')
    parser.add_argument('--adapter_path', type=str, default=None,
                        help='Local path to adapter weights (overrides index config)')

    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    # Mutually exclusive check
    if args.adapter_name and args.adapter_path:
        parser.error("Cannot specify both --adapter_name and --adapter_path. Choose one.")

    return args


def main():
    """Main function"""
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    dataset_name = os.path.basename(args.data_dir)
    print("=" * 60)
    print(f"SAPBERT BC5CDR-{dataset_name.title()} Evaluation (Adapter-aware)")
    print("=" * 60)

    try:
        # Initialize evaluator with adapter overrides (if any)
        evaluator = BC5CDREvaluator(
            index_path=args.index_path,
            data_dir=args.data_dir,
            adapter_name=args.adapter_name,
            adapter_path=args.adapter_path
        )

        # Run evaluation with topk parameter
        results = evaluator.run_evaluation(topk=args.topk)

        # Print summary
        rcfg = results.get('runtime_config', {})
        print(f"\n⚙️  RUNTIME CONFIG")
        print(f"Model: {rcfg.get('model_name')}")
        print(f"Index Type: {rcfg.get('index_type')}")
        print(f"Max Length: {rcfg.get('max_length')}")
        print(f"Adapter: {'ON' if rcfg.get('use_adapter_effective') else 'OFF'}")
        if rcfg.get('use_adapter_effective'):
            if rcfg.get('adapter_path_effective'):
                print(f"Adapter (Local): {rcfg.get('adapter_path_effective')}")
            else:
                print(f"Adapter (Hub): {rcfg.get('adapter_name_effective')}")

        # Print dataset statistics
        if 'dataset_statistics' in results:
            ds = results['dataset_statistics']
            print(f"\n📈 DATASET STATISTICS (count_bc5cdr_mentions.py methodology)")
            print(f"Raw mentions in files: {ds['total_raw_mentions']:,}")
            print(f"Unique mentions (case-insensitive): {ds['unique_mentions']:,}")
            print(f"Unique CUIs: {ds['unique_cuis']:,}")
            print(f"\n✅ EVALUATION APPROACH")
            print(f"Total queries evaluated: {ds['total_queries']:,}")
            print(f"Queries == Raw mentions: {ds['queries_equal_mentions']}")
            print(f"Method: Each individual mention evaluated separately (NO deduplication)")

        print(f"\n📊 EVALUATION SUMMARY")
        print(f"Total Queries Evaluated: {results['total_queries']:,}")
        print(f"Correct Predictions: {results['correct_predictions']:,}")
        print(f"Top-1 Accuracy: {results['top1_accuracy']:.4f}")
        print(f"Error Rate: {results['error_analysis']['error_rate']:.4f}")

        # Print acc@k metrics
        print(f"\n📈 ACCURACY@K METRICS")
        acc_metrics = []
        for i in range(1, 11):
            acc_key = f'acc{i}'
            if acc_key in results:
                acc_metrics.append(f"acc@{i}: {results[acc_key]:.4f}")

        if acc_metrics:
            # Print in rows of 5 for readability
            for i in range(0, len(acc_metrics), 5):
                print("  " + "  |  ".join(acc_metrics[i:i+5]))

        print(f"\n🔍 ERROR TYPE DISTRIBUTION")
        for error_type, count in results['error_analysis']['error_counts'].items():
            percentage = (count / len(results['error_cases'])) * 100 if results['error_cases'] else 0
            print(f"  {error_type}: {count} ({percentage:.1f}%)")

        print(f"\n📝 GENERATING REPORTS")
        error_report_path = evaluator.generate_error_report(args.output_dir)
        results_path = evaluator.save_results(args.output_dir)

        # Write error cases to CSV (Golden CUI and Query Text)
        error_cases_df = pd.DataFrame(results['error_cases'])
        csv_path = os.path.join(args.output_dir, f'{dataset_name}_error_cases_golden_cui_query.csv')
        # Rename 'mention' to 'query_text' for CSV output
        error_cases_df.rename(columns={'mention': 'query_text'}, inplace=True)
        error_cases_df[['golden_cui', 'query_text']].to_csv(csv_path, index=False)
        print(f"Error cases CSV written to: {csv_path}")

        print(f"Error analysis report: {error_report_path}")
        print(f"Detailed results: {results_path}")
        print(f"\n✅ Evaluation completed successfully!")

    except KeyboardInterrupt:
        print(f"\n⚠️  Evaluation cancelled by user.")
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
