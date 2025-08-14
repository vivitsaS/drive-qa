#!/usr/bin/env python3
"""
RAG Agent Evaluation Script

This script runs the RAG agent on multiple scenes, keyframes, and QA types,
then evaluates the performance using various metrics including semantic similarity,
exact match, and LLM as a judge.
"""

import argparse
import os
import sys
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import time
from loguru import logger

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag.rag_agent import RAGAgent
from parsers.data_loader import DataLoader


def setup_logging():
    """Configure logging for the evaluation"""
    log_file = f"logs/rag_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(log_file, rotation="10 MB", level="INFO")
    logger.info("Starting RAG Agent Evaluation")


def validate_api_key() -> str:
    """Validate and return the API key"""
    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GEMINI_KEY')
    if not api_key:
        logger.error("Please set GEMINI_API_KEY environment variable")
        logger.info("You can get an API key from: https://makersuite.google.com/app/apikey")
        sys.exit(1)
    return api_key


def get_evaluation_config() -> Dict[str, Any]:
    """Get the evaluation configuration with scenes, keyframes, and QA types"""
    data_loader = DataLoader()
    
    config = {
        'scenes': list(range(1, 7)),  # Scenes 1-6
        'qa_types': ['perception', 'planning', 'prediction', 'behavior'],
        'max_keyframes_per_scene': 3,  # Limit to avoid too many evaluations
        'max_qa_pairs_per_type': 5     # Limit to avoid too many evaluations
    }
    
    # Get actual keyframe counts per scene
    scene_keyframe_counts = {}
    for scene_id in config['scenes']:
        scene_info = data_loader.get_keyframe_info_for_scene(scene_id)
        scene_keyframe_counts[scene_id] = len(scene_info['keyframe_tokens'])
    
    config['scene_keyframe_counts'] = scene_keyframe_counts
    return config


def generate_evaluation_tasks(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate a list of evaluation tasks"""
    tasks = []
    data_loader = DataLoader()
    
    for scene_id in config['scenes']:
        max_keyframes = min(config['max_keyframes_per_scene'], 
                           config['scene_keyframe_counts'][scene_id])
        
        for keyframe_id in range(1, max_keyframes + 1):
            # Load scene data to get QA pair counts
            scene_data = data_loader.load_scene_data(scene_id)
            keyframe_token = data_loader._assign_keyframe_token(scene_id, keyframe_id)
            
            if keyframe_token not in scene_data['key_frames']:
                continue
                
            keyframe_data = scene_data['key_frames'][keyframe_token]
            qa_data = keyframe_data.get('QA', {})
            
            for qa_type in config['qa_types']:
                if qa_type not in qa_data:
                    continue
                    
                qa_pairs = qa_data[qa_type]
                max_qa_pairs = min(config['max_qa_pairs_per_type'], len(qa_pairs))
                
                for qa_serial in range(1, max_qa_pairs + 1):
                    tasks.append({
                        'scene_id': scene_id,
                        'keyframe_id': keyframe_id,
                        'qa_type': qa_type,
                        'qa_serial': qa_serial
                    })
    
    return tasks


def run_single_evaluation(agent: RAGAgent, task: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single evaluation task"""
    logger.info(f"Evaluating: Scene {task['scene_id']}, Keyframe {task['keyframe_id']}, "
                f"QA Type {task['qa_type']}, Serial {task['qa_serial']}")
    
    start_time = time.time()
    
    try:
        result = agent.answer_question(
            scene_id=task['scene_id'],
            keyframe_id=task['keyframe_id'],
            qa_type=task['qa_type'],
            qa_serial=task['qa_serial']
        )
        
        evaluation_time = time.time() - start_time
        
        if result["success"]:
            return {
                'task': task,
                'success': True,
                'question': result['question'],
                'model_answer': result['model_answer'],
                'ground_truth_answer': result['ground_truth_answer'],
                'metadata': result['metadata'],
                'evaluation_time': evaluation_time,
                'error': None
            }
        else:
            return {
                'task': task,
                'success': False,
                'question': None,
                'model_answer': None,
                'ground_truth_answer': None,
                'metadata': result.get('metadata', {}),
                'evaluation_time': evaluation_time,
                'error': result.get('error', 'Unknown error')
            }
            
    except Exception as e:
        evaluation_time = time.time() - start_time
        logger.error(f"Error in evaluation: {e}")
        return {
            'task': task,
            'success': False,
            'question': None,
            'model_answer': None,
            'ground_truth_answer': None,
            'metadata': {},
            'evaluation_time': evaluation_time,
            'error': str(e)
        }


def calculate_exact_match(model_answer: str, ground_truth: str) -> bool:
    """Calculate exact match between model answer and ground truth"""
    return model_answer.strip().lower() == ground_truth.strip().lower()


def calculate_semantic_similarity(model_answer: str, ground_truth: str) -> float:
    """Calculate semantic similarity using simple word overlap"""
    model_words = set(model_answer.strip().lower().split())
    gt_words = set(ground_truth.strip().lower().split())
    
    if not model_words or not gt_words:
        return 0.0
    
    intersection = model_words.intersection(gt_words)
    union = model_words.union(gt_words)
    
    return len(intersection) / len(union) if union else 0.0


def save_results_to_csv(results: List[Dict[str, Any]], output_file: str):
    """Save evaluation results to CSV file"""
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'scene_id', 'keyframe_id', 'qa_type', 'qa_serial',
            'success', 'question', 'model_answer', 'ground_truth_answer',
            'exact_match', 'semantic_similarity', 'evaluation_time', 'error'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            row = {
                'scene_id': result['task']['scene_id'],
                'keyframe_id': result['task']['keyframe_id'],
                'qa_type': result['task']['qa_type'],
                'qa_serial': result['task']['qa_serial'],
                'success': result['success'],
                'question': result.get('question', ''),
                'model_answer': result.get('model_answer', ''),
                'ground_truth_answer': result.get('ground_truth_answer', ''),
                'evaluation_time': result.get('evaluation_time', 0),
                'error': result.get('error', '')
            }
            
            # Calculate metrics if successful
            if result['success'] and result['model_answer'] and result['ground_truth_answer']:
                row['exact_match'] = calculate_exact_match(result['model_answer'], result['ground_truth_answer'])
                row['semantic_similarity'] = calculate_semantic_similarity(result['model_answer'], result['ground_truth_answer'])
            else:
                row['exact_match'] = False
                row['semantic_similarity'] = 0.0
            
            writer.writerow(row)


def generate_summary_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a summary report of the evaluation results"""
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    total_tasks = len(results)
    total_successful = len(successful_results)
    total_failed = len(failed_results)
    
    # Calculate metrics for successful evaluations
    exact_matches = 0
    semantic_similarities = []
    evaluation_times = []
    
    for result in successful_results:
        if result['model_answer'] and result['ground_truth_answer']:
            if calculate_exact_match(result['model_answer'], result['ground_truth_answer']):
                exact_matches += 1
            semantic_similarities.append(
                calculate_semantic_similarity(result['model_answer'], result['ground_truth_answer'])
            )
        evaluation_times.append(result['evaluation_time'])
    
    # Calculate averages
    avg_semantic_similarity = sum(semantic_similarities) / len(semantic_similarities) if semantic_similarities else 0
    avg_evaluation_time = sum(evaluation_times) / len(evaluation_times) if evaluation_times else 0
    
    # Per QA type analysis
    qa_type_stats = {}
    for qa_type in ['perception', 'planning', 'prediction', 'behavior']:
        type_results = [r for r in successful_results if r['task']['qa_type'] == qa_type]
        if type_results:
            type_exact_matches = sum(1 for r in type_results 
                                   if r['model_answer'] and r['ground_truth_answer'] 
                                   and calculate_exact_match(r['model_answer'], r['ground_truth_answer']))
            type_similarities = [calculate_semantic_similarity(r['model_answer'], r['ground_truth_answer']) 
                               for r in type_results if r['model_answer'] and r['ground_truth_answer']]
            
            qa_type_stats[qa_type] = {
                'count': len(type_results),
                'exact_match_rate': type_exact_matches / len(type_results) if type_results else 0,
                'avg_semantic_similarity': sum(type_similarities) / len(type_similarities) if type_similarities else 0
            }
    
    return {
        'total_tasks': total_tasks,
        'total_successful': total_successful,
        'total_failed': total_failed,
        'success_rate': total_successful / total_tasks if total_tasks > 0 else 0,
        'exact_match_rate': exact_matches / total_successful if total_successful > 0 else 0,
        'avg_semantic_similarity': avg_semantic_similarity,
        'avg_evaluation_time': avg_evaluation_time,
        'qa_type_stats': qa_type_stats,
        'error_summary': {
            'total_errors': total_failed,
            'common_errors': {}
        }
    }


def save_summary_report(summary: Dict[str, Any], output_file: str):
    """Save summary report to JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main():
    """Main entry point for the evaluation script"""
    parser = argparse.ArgumentParser(description="Evaluate the RAG Agent")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", 
                       help="Directory to save evaluation results")
    parser.add_argument("--max-scenes", type=int, default=6, 
                       help="Maximum number of scenes to evaluate")
    parser.add_argument("--max-keyframes", type=int, default=3, 
                       help="Maximum keyframes per scene to evaluate")
    parser.add_argument("--max-qa-pairs", type=int, default=5, 
                       help="Maximum QA pairs per type to evaluate")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Validate API key
    api_key = validate_api_key()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get evaluation configuration
    config = get_evaluation_config()
    config['max_keyframes_per_scene'] = args.max_keyframes
    config['max_qa_pairs_per_type'] = args.max_qa_pairs
    config['scenes'] = config['scenes'][:args.max_scenes]
    
    logger.info(f"Evaluation configuration: {config}")
    
    # Generate evaluation tasks
    tasks = generate_evaluation_tasks(config)
    logger.info(f"Generated {len(tasks)} evaluation tasks")
    
    # Initialize RAG agent
    agent = RAGAgent(api_key)
    
    # Run evaluations
    results = []
    for i, task in enumerate(tasks, 1):
        logger.info(f"Progress: {i}/{len(tasks)}")
        result = run_single_evaluation(agent, task)
        results.append(result)
        
        # Add a small delay to avoid rate limiting
        time.sleep(0.5)
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_file = output_dir / f"evaluation_results_{timestamp}.csv"
    save_results_to_csv(results, csv_file)
    logger.info(f"Detailed results saved to {csv_file}")
    
    # Generate and save summary report
    summary = generate_summary_report(results)
    summary_file = output_dir / f"evaluation_summary_{timestamp}.json"
    save_summary_report(summary, summary_file)
    logger.info(f"Summary report saved to {summary_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Total tasks: {summary['total_tasks']}")
    print(f"Successful: {summary['total_successful']}")
    print(f"Failed: {summary['total_failed']}")
    print(f"Success rate: {summary['success_rate']:.2%}")
    print(f"Exact match rate: {summary['exact_match_rate']:.2%}")
    print(f"Average semantic similarity: {summary['avg_semantic_similarity']:.3f}")
    print(f"Average evaluation time: {summary['avg_evaluation_time']:.2f}s")
    
    print("\nPer QA Type Performance:")
    for qa_type, stats in summary['qa_type_stats'].items():
        print(f"  {qa_type}: {stats['count']} tasks, "
              f"exact match: {stats['exact_match_rate']:.2%}, "
              f"semantic similarity: {stats['avg_semantic_similarity']:.3f}")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

