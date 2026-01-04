#!/usr/bin/env python3
"""
SAPBERT Evaluation Script for BC5CDR Dataset (Chemical and Disease)

Now adapter-aware:
- Respects adapter settings saved with the index
- Optional CLI overrides: --adapter_name or --adapter_path (mutually exclusive)

Usage:
    python eval_sapbert_bc5cdr_fixed.py --index_path ./utils/NEL/UMLS/indexes/Wikidata --data_dir ./data/bc5cdr-chemical
    python eval_sapbert_bc5cdr_fixed.py --index_path ./utils/NEL/UMLS/indexes/Wikidata --data_dir ./data/bc5cdr-disease
    # Override adapter from Hub
    python eval_sapbert_bc5cdr_fixed.py --index_path ... --data_dir ... --adapter_name username/my-adapter
    # Override adapter from local path
    python eval_sapbert_bc5cdr_fixed.py --index_path ... --data_dir ... --adapter_path ./my_adapter
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

        Returns:
            List of test query dictionaries
        """
        processed_test_dir = os.path.join(self.data_dir, 'processed_test')
        logger.info(f"Loading test queries from: {processed_test_dir}")

        queries = []
        concept_files = [f for f in os.listdir(processed_test_dir) if f.endswith('.concept')]
        logger.info(f"Found {len(concept_files)} concept files to process")

        for concept_file in tqdm(concept_files, desc="Loading test queries"):
            file_path = os.path.join(processed_test_dir, concept_file)

            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Group mentions by document and CUI
            doc_mentions = defaultdict(list)
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('||')
                if len(parts) == 5:  # doc_id||start_pos|end_pos||entity_type||mention_text||cui
                    doc_id = parts[0]
                    mention_text = parts[3]  # Index 3 is the entity name
                    cui = parts[4]  # Index 4 is the matching CUI

                    # Process all entities regardless of type
                    doc_mentions[doc_id].append({
                        'mention': mention_text,
                        'cui': cui
                    })
                else:
                    continue

            # Create queries for each document
            for doc_id, mentions in doc_mentions.items():
                # Group mentions by CUI
                cui_mentions = defaultdict(list)
                for mention in mentions:
                    cui_mentions[mention['cui']].append(mention['mention'])

                # Create a query for each unique CUI
                for cui, mention_texts in cui_mentions.items():
                    # Use the first mention as the query text
                    query_text = mention_texts[0]
                    queries.append({
                        'doc_id': doc_id,
                        'query_text': query_text,
                        'golden_cui': cui,
                        'all_mentions': mention_texts,
                        'concept_file': concept_file
                    })

        logger.info(f"Loaded {len(queries)} test queries")
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

    def evaluate_single_query(self, query: Dict) -> Dict:
        """
        Evaluate a single query

        Args:
            query: Query dictionary with query_text and golden_cui

        Returns:
            Evaluation result dictionary
        """
        query_text = query['query_text']
        golden_cui = query['golden_cui']

        # Search for top-1 result
        search_results = self.searcher.search(query_text, k=1, return_scores=True)

        if not search_results:
            return {
                'query': query,
                'predicted_cui': None,
                'predicted_name': None,
                'similarity_score': 0.0,
                'is_correct': False,
                'error_type': 'no_results',
                'search_results': []
            }

        top_result = search_results[0]
        predicted_cui = top_result.get('entity_id', '')
        predicted_name = top_result.get('aliases', [''])[0] if top_result.get('aliases') else ''
        similarity_score = top_result.get('similarity_score', 0.0)

        # Check if prediction is correct - simplified CUI matching
        is_correct = self.check_cui_match(predicted_cui, golden_cui)

        error_type = None
        if not is_correct:
            error_type = self.categorize_error(predicted_cui, golden_cui, query_text, predicted_name)

        return {
            'query': query,
            'predicted_cui': predicted_cui,
            'predicted_name': predicted_name,
            'similarity_score': similarity_score,
            'is_correct': is_correct,
            'error_type': error_type,
            'search_results': search_results
        }

    def check_cui_match(self, predicted_cui: str, golden_cui: str) -> bool:
        """
        Check if predicted CUI matches golden CUI.
        Handles cases where the golden CUI may contain multiple pipe-separated IDs.
        """
        if not predicted_cui or not golden_cui:
            return False
        golden_cuis = golden_cui.split('|')
        return predicted_cui in golden_cuis

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

    def run_evaluation(self) -> Dict:
        """
        Run the complete evaluation
        """
        logger.info(f"Starting BC5CDR-{self.dataset_name} evaluation")

        # Load data
        self.test_queries = self.load_test_queries()

        # Initialize searcher
        self.initialize_searcher()

        # Run evaluation
        logger.info(f"Evaluating {len(self.test_queries)} queries")

        correct_predictions = 0
        error_cases = []
        error_counts = Counter()
        similarity_scores = []

        for query in tqdm(self.test_queries, desc="Evaluating queries"):
            result = self.evaluate_single_query(query)

            if result['is_correct']:
                correct_predictions += 1
            else:
                error_cases.append(result)
                if result['error_type']:
                    error_counts[result['error_type']] += 1

            similarity_scores.append(result['similarity_score'])

        total_queries = len(self.test_queries)
        top1_accuracy = correct_predictions / total_queries if total_queries > 0 else 0.0

        self.results.update({
            'total_queries': total_queries,
            'correct_predictions': correct_predictions,
            'top1_accuracy': top1_accuracy,
            'error_cases': error_cases,
            'error_analysis': {
                'error_counts': dict(error_counts),
                'error_rate': len(error_cases) / total_queries if total_queries > 0 else 0.0,
                'avg_similarity_score': np.mean(similarity_scores) if similarity_scores else 0.0,
                'min_similarity_score': np.min(similarity_scores) if similarity_scores else 0.0,
                'max_similarity_score': np.max(similarity_scores) if similarity_scores else 0.0
            },
            'performance_metrics': {
                'total_queries': total_queries,
                'correct_predictions': correct_predictions,
                'incorrect_predictions': len(error_cases),
                'top1_accuracy': top1_accuracy,
                'error_rate': len(error_cases) / total_queries if total_queries > 0 else 0.0
            }
        })

        logger.info(f"Evaluation completed. Top-1 Accuracy: {top1_accuracy:.4f}")
        return self.results

    def generate_error_report(self, output_dir: str = None) -> str:
        """
        Generate detailed error analysis report
        """
        if output_dir is None:
            output_dir = os.path.dirname(__file__)

        error_data = []
        for error_case in self.results['error_cases']:
            query = error_case['query']
            error_data.append({
                'doc_id': query['doc_id'],
                'query_text': query['query_text'],
                'golden_cui': query['golden_cui'],
                'predicted_cui': error_case['predicted_cui'],
                'predicted_name': error_case['predicted_name'],
                'similarity_score': error_case['similarity_score'],
                'error_type': error_case['error_type'],
                'concept_file': query['concept_file']
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
                query = error_case['query']
                f.write(f"\nError Case {i}:\n")
                f.write(f"  Document ID: {query['doc_id']}\n")
                f.write(f"  Query Text: '{query['query_text']}'\n")
                f.write(f"  Golden CUI: {query['golden_cui']}\n")
                f.write(f"  Predicted CUI: {error_case['predicted_cui']}\n")
                f.write(f"  Predicted Name: '{error_case['predicted_name']}'\n")
                f.write(f"  Similarity Score: {error_case['similarity_score']:.4f}\n")
                f.write(f"  Error Type: {error_case['error_type']}\n")
                f.write(f"  Concept File: {query['concept_file']}\n")

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

        # Run evaluation
        results = evaluator.run_evaluation()

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

        print(f"\n📊 EVALUATION SUMMARY")
        print(f"Total Queries: {results['total_queries']}")
        print(f"Correct Predictions: {results['correct_predictions']}")
        print(f"Top-1 Accuracy: {results['top1_accuracy']:.4f}")
        print(f"Error Rate: {results['error_analysis']['error_rate']:.4f}")

        print(f"\n🔍 ERROR TYPE DISTRIBUTION")
        for error_type, count in results['error_analysis']['error_counts'].items():
            percentage = (count / len(results['error_cases'])) * 100 if results['error_cases'] else 0
            print(f"  {error_type}: {count} ({percentage:.1f}%)")

        print(f"\n📝 GENERATING REPORTS")
        error_report_path = evaluator.generate_error_report(args.output_dir)
        results_path = evaluator.save_results(args.output_dir)

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
