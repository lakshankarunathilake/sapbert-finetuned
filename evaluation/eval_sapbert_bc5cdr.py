#!/usr/bin/env python3
"""
SAPBERT Evaluation Script for BC5CDR-Chemical Dataset

This script evaluates SAPBERT model performance on the BC5CDR-Chemical dataset
using the search_sapbert_index functionality. It focuses on top-1 accuracy
and provides detailed error analysis.

Usage:
    python eval_sapbert_bc5cdr.py --index_path ./utils/NEL/UMLS/indexes/Wikidata --data_dir ./data/bc5cdr-chemical

Features:
- Top-1 accuracy evaluation
- Detailed error case analysis
- Performance metrics
- Error categorization
- Export of results and error cases
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

# Add the utils directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils', 'NEL', 'UMLS'))

from search_sapbert_index import SAPBERTIndexSearcher

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BC5CDREvaluator:
    """
    Evaluator for BC5CDR-Chemical dataset using SAPBERT search functionality
    """
    
    def __init__(self, index_path: str, data_dir: str):
        """
        Initialize the evaluator
        
        Args:
            index_path: Path to the SAPBERT index files (without extension)
            data_dir: Path to the BC5CDR-Chemical data directory
        """
        self.index_path = index_path
        self.data_dir = data_dir
        self.searcher = None
        self.dictionary = {}
        self.test_queries = []
        self.results = {
            'total_queries': 0,
            'correct_predictions': 0,
            'top1_accuracy': 0.0,
            'error_cases': [],
            'error_analysis': {},
            'performance_metrics': {}
        }
        
    def load_dictionary(self) -> Dict[str, str]:
        """
        Load the BC5CDR-Chemical dictionary
        
        Returns:
            Dictionary mapping entity names to CUIs
        """
        dict_path = os.path.join(self.data_dir, 'test_dictionary.txt')
        logger.info(f"Loading dictionary from: {dict_path}")
        
        dictionary = {}
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('||')
                if len(parts) >= 2:
                    cui = parts[0]
                    name = parts[1]
                    dictionary[name] = cui
                    
        logger.info(f"Loaded {len(dictionary)} dictionary entries")
        return dictionary
    
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
            print(f"Read {len(lines)} lines from {file_path}")
                
            # Group mentions by document and CUI
            doc_mentions = defaultdict(list)
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split('||')
                if len(parts) == 5:  # doc_id||start_pos|end_pos||entity_type||mention_text||cui
                    doc_id = parts[0]
                    # Split start_pos|end_pos
                    pos_parts = parts[1].split('|')
                    if len(pos_parts) >= 2:
                        start_pos = pos_parts[0]
                        end_pos = pos_parts[1]
                    else:
                        continue
                    entity_type = parts[2]
                    mention_text = parts[3]  # Index 3 is the entity name
                    cui = parts[4]          # Index 4 is the matching CUI
                else:
                    print(f"Skipping line: {line}")
                    continue
                
                # Debug logging for first few lines
                if len(queries) < 5:
                    logger.debug(f"Parsed line: doc_id={doc_id}, entity_type={entity_type}, mention_text='{mention_text}', cui={cui}")
                
                doc_mentions[doc_id].append({
                    'mention': mention_text,
                    'cui': cui,
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'entity_type': entity_type
                })
            
            # Create queries for each document
            for doc_id, mentions in doc_mentions.items():
                # Group mentions by CUI
                cui_mentions = defaultdict(list)
                for mention in mentions:
                    cui_mentions[mention['cui']].append(mention['mention'])
                
                # Debug logging for first few documents
                if len(queries) < 5:
                    logger.debug(f"Document {doc_id} has {len(mentions)} mentions, {len(cui_mentions)} unique CUIs")
                
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
        """Initialize the SAPBERT searcher"""
        logger.info(f"Initializing SAPBERT searcher with index: {self.index_path}")
        self.searcher = SAPBERTIndexSearcher()
        config = self.searcher.load_index(self.index_path)
        logger.info(f"Index loaded successfully. Model: {config.get('model_name', 'Unknown')}")
    
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
        
        # Check if prediction is correct
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
        Check if predicted CUI matches golden CUI
        
        Args:
            predicted_cui: Predicted entity ID
            golden_cui: Golden standard CUI
            
        Returns:
            True if match, False otherwise
        """
        if not predicted_cui or not golden_cui:
            return False
        
        # Direct match
        if predicted_cui == golden_cui:
            return True
        
        # Check if predicted CUI is in the dictionary and maps to golden CUI
        if predicted_cui in self.dictionary:
            return self.dictionary[predicted_cui] == golden_cui
        
        return False
    
    def categorize_error(self, predicted_cui: str, golden_cui: str, query_text: str, predicted_name: str) -> str:
        """
        Categorize the type of error
        
        Args:
            predicted_cui: Predicted entity ID
            golden_cui: Golden standard CUI
            query_text: Original query text
            predicted_name: Predicted entity name
            
        Returns:
            Error category string
        """
        if not predicted_cui:
            return 'no_prediction'
        
        # Check if it's a semantic similarity issue
        if predicted_name and query_text.lower() in predicted_name.lower():
            return 'partial_match'
        
        # Check if it's a completely different entity
        if predicted_cui != golden_cui:
            return 'wrong_entity'
        
        return 'unknown_error'
    
    def run_evaluation(self) -> Dict:
        """
        Run the complete evaluation
        
        Returns:
            Evaluation results dictionary
        """
        logger.info("Starting BC5CDR-Chemical evaluation")
        
        # Load data
        self.dictionary = self.load_dictionary()
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
        
        # Calculate metrics
        total_queries = len(self.test_queries)
        top1_accuracy = correct_predictions / total_queries if total_queries > 0 else 0.0
        
        # Update results
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
        
        Args:
            output_dir: Directory to save the report (default: evaluation directory)
            
        Returns:
            Path to the generated report file
        """
        if output_dir is None:
            output_dir = os.path.dirname(__file__)
        
        # Create error analysis DataFrame
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
        
        # Generate report
        report_path = os.path.join(output_dir, 'bc5cdr_chemical_error_analysis.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("BC5CDR-Chemical SAPBERT Evaluation Error Analysis\n")
            f.write("=" * 60 + "\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Queries: {self.results['total_queries']}\n")
            f.write(f"Correct Predictions: {self.results['correct_predictions']}\n")
            f.write(f"Incorrect Predictions: {len(self.results['error_cases'])}\n")
            f.write(f"Top-1 Accuracy: {self.results['top1_accuracy']:.4f}\n")
            f.write(f"Error Rate: {self.results['error_analysis']['error_rate']:.4f}\n\n")
            
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
            for i, error_case in enumerate(self.results['error_cases'][:50], 1):  # Show first 50 errors
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
        
        Args:
            output_dir: Directory to save the results (default: evaluation directory)
            
        Returns:
            Path to the saved results file
        """
        if output_dir is None:
            output_dir = os.path.dirname(__file__)
        
        results_path = os.path.join(output_dir, 'bc5cdr_chemical_evaluation_results.json')
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Create a serializable version of results
        serializable_results = json.loads(json.dumps(self.results, default=convert_numpy))
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation results saved to: {results_path}")
        return results_path


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate SAPBERT on BC5CDR-Chemical dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python eval_sapbert_bc5cdr.py --index_path ./utils/NEL/UMLS/indexes/Wikidata --data_dir ./data/bc5cdr-chemical

  # Evaluation with custom output directory
  python eval_sapbert_bc5cdr.py --index_path ./utils/NEL/UMLS/indexes/Wikidata --data_dir ./data/bc5cdr-chemical --output_dir ./results

  # Evaluation with verbose logging
  python eval_sapbert_bc5cdr.py --index_path ./utils/NEL/UMLS/indexes/Wikidata --data_dir ./data/bc5cdr-chemical --verbose
        """
    )
    
    parser.add_argument(
        '--index_path',
        type=str,
        required=True,
        help='Path to SAPBERT index files (without extension)'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to BC5CDR-Chemical data directory'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save results and reports (default: evaluation directory)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("=" * 60)
    print("SAPBERT BC5CDR-Chemical Evaluation")
    print("=" * 60)
    
    try:
        # Initialize evaluator
        evaluator = BC5CDREvaluator(
            index_path=args.index_path,
            data_dir=args.data_dir
        )
        
        # Run evaluation
        results = evaluator.run_evaluation()
        
        # Print summary
        print(f"\n📊 EVALUATION SUMMARY")
        print(f"Total Queries: {results['total_queries']}")
        print(f"Correct Predictions: {results['correct_predictions']}")
        print(f"Top-1 Accuracy: {results['top1_accuracy']:.4f}")
        print(f"Error Rate: {results['error_analysis']['error_rate']:.4f}")
        
        # Print error type distribution
        print(f"\n🔍 ERROR TYPE DISTRIBUTION")
        for error_type, count in results['error_analysis']['error_counts'].items():
            percentage = (count / len(results['error_cases'])) * 100 if results['error_cases'] else 0
            print(f"  {error_type}: {count} ({percentage:.1f}%)")
        
        # Generate reports
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
