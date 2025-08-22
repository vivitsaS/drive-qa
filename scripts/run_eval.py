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
from typing import Dict, List, Any
from datetime import datetime
import time
from loguru import logger
import google.generativeai as genai

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
            # Add LLM judge evaluation
            llm_evaluation = calculate_llm_judge_score(
                result['model_answer'], 
                result['ground_truth_answer'], 
                result['question']
            )
            
            return {
                'task': task,
                'success': True,
                'question': result['question'],
                'model_answer': result['model_answer'],
                'ground_truth_answer': result['ground_truth_answer'],
                'metadata': result['metadata'],
                'evaluation_time': evaluation_time,
                'llm_evaluation': llm_evaluation,
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
                'llm_evaluation': {'success': False, 'error': 'RAG agent failed'},
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
            'llm_evaluation': {'success': False, 'error': str(e)},
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


def calculate_llm_judge_score(model_answer: str, ground_truth: str, question: str) -> Dict[str, Any]:
    """
    Use LLM to evaluate semantic correctness of the model answer.
    
    Args:
        model_answer: The RAG agent's answer
        ground_truth: The ground truth answer
        question: The original question
        
    Returns:
        Dictionary with evaluation scores and reasoning
    """
    try:
        # Create evaluation prompt for Gemini
        evaluation_prompt = f"""
You are an expert evaluator for autonomous driving question-answering systems. Evaluate the semantic correctness of the model's answer compared to the ground truth.

Question: {question}
Ground Truth Answer: {ground_truth}
Model Answer: {model_answer}

Evaluate the model's answer on these criteria:
1. **Semantic Correctness** (0-10): Does the model's answer convey the same core meaning as the ground truth?
2. **Completeness** (0-10): Does the model provide sufficient detail and context?
3. **Accuracy** (0-10): Are the facts stated in the model's answer accurate?
4. **Usefulness** (0-10): Is the answer helpful for autonomous driving decision-making?

Consider that:
- Detailed explanations are better than simple answers
- Contextual reasoning is valuable
- The model may provide more comprehensive information than ground truth
- Focus on whether the core meaning and intent are correct

Provide your evaluation as JSON:
{{
    "semantic_correctness": <score>,
    "completeness": <score>,
    "accuracy": <score>,
    "usefulness": <score>,
    "overall_score": <average of all scores>,
    "reasoning": "<explanation of your evaluation>"
}}
"""
        
        # Configure Gemini API
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GEMINI_KEY')
        if not api_key:
            return {
                "success": False,
                "error": "No API key found",
                "raw_response": ""
            }
        
        genai.configure(api_key=api_key)
        
        # Use the same model for evaluation
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(evaluation_prompt)
        
        # Parse the JSON response
        try:
            import json
            evaluation = json.loads(response.text)
            return {
                "success": True,
                "scores": evaluation,
                "raw_response": response.text
            }
        except json.JSONDecodeError:
            # Fallback: extract scores from text
            logger.warning(f"Failed to parse JSON response: {response.text}")
            return {
                "success": False,
                "error": "Failed to parse JSON response",
                "raw_response": response.text
            }
            
    except Exception as e:
        logger.error(f"Error in LLM judge evaluation: {e}")
        return {
            "success": False,
            "error": str(e),
            "raw_response": ""
        }


def save_results_to_csv(results: List[Dict[str, Any]], output_file: str):
    """Save evaluation results to CSV file"""
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'scene_id', 'keyframe_id', 'qa_type', 'qa_serial',
            'success', 'question', 'model_answer', 'ground_truth_answer',
            'exact_match', 'semantic_similarity', 'llm_semantic_correctness', 
            'llm_completeness', 'llm_accuracy', 'llm_usefulness', 'llm_overall_score',
            'evaluation_time', 'error'
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
                
                # Add LLM judge scores
                llm_eval = result.get('llm_evaluation', {})
                if llm_eval.get('success'):
                    scores = llm_eval['scores']
                    row['llm_semantic_correctness'] = scores.get('semantic_correctness', 0)
                    row['llm_completeness'] = scores.get('completeness', 0)
                    row['llm_accuracy'] = scores.get('accuracy', 0)
                    row['llm_usefulness'] = scores.get('usefulness', 0)
                    row['llm_overall_score'] = scores.get('overall_score', 0)
                else:
                    row['llm_semantic_correctness'] = 0
                    row['llm_completeness'] = 0
                    row['llm_accuracy'] = 0
                    row['llm_usefulness'] = 0
                    row['llm_overall_score'] = 0
            else:
                row['exact_match'] = False
                row['semantic_similarity'] = 0.0
                row['llm_semantic_correctness'] = 0
                row['llm_completeness'] = 0
                row['llm_accuracy'] = 0
                row['llm_usefulness'] = 0
                row['llm_overall_score'] = 0
            
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
    
    # LLM judge metrics
    llm_semantic_correctness = []
    llm_completeness = []
    llm_accuracy = []
    llm_usefulness = []
    llm_overall_scores = []
    
    for result in successful_results:
        if result['model_answer'] and result['ground_truth_answer']:
            if calculate_exact_match(result['model_answer'], result['ground_truth_answer']):
                exact_matches += 1
            semantic_similarities.append(
                calculate_semantic_similarity(result['model_answer'], result['ground_truth_answer'])
            )
        evaluation_times.append(result['evaluation_time'])
        
        # Collect LLM judge scores
        llm_eval = result.get('llm_evaluation', {})
        if llm_eval.get('success') and 'scores' in llm_eval:
            scores = llm_eval['scores']
            llm_semantic_correctness.append(scores.get('semantic_correctness', 0))
            llm_completeness.append(scores.get('completeness', 0))
            llm_accuracy.append(scores.get('accuracy', 0))
            llm_usefulness.append(scores.get('usefulness', 0))
            llm_overall_scores.append(scores.get('overall_score', 0))
    
    # Calculate averages
    avg_semantic_similarity = sum(semantic_similarities) / len(semantic_similarities) if semantic_similarities else 0
    avg_evaluation_time = sum(evaluation_times) / len(evaluation_times) if evaluation_times else 0
    
    # LLM judge averages
    avg_llm_semantic_correctness = sum(llm_semantic_correctness) / len(llm_semantic_correctness) if llm_semantic_correctness else 0
    avg_llm_completeness = sum(llm_completeness) / len(llm_completeness) if llm_completeness else 0
    avg_llm_accuracy = sum(llm_accuracy) / len(llm_accuracy) if llm_accuracy else 0
    avg_llm_usefulness = sum(llm_usefulness) / len(llm_usefulness) if llm_usefulness else 0
    avg_llm_overall = sum(llm_overall_scores) / len(llm_overall_scores) if llm_overall_scores else 0
    
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
            
            # LLM judge scores for this QA type
            type_llm_scores = [r.get('llm_evaluation', {}).get('scores', {}).get('overall_score', 0) 
                              for r in type_results if r.get('llm_evaluation', {}).get('success')]
            
            qa_type_stats[qa_type] = {
                'count': len(type_results),
                'exact_match_rate': type_exact_matches / len(type_results) if type_results else 0,
                'avg_semantic_similarity': sum(type_similarities) / len(type_similarities) if type_similarities else 0,
                'avg_llm_score': sum(type_llm_scores) / len(type_llm_scores) if type_llm_scores else 0
            }
    
    return {
        'total_tasks': total_tasks,
        'total_successful': total_successful,
        'total_failed': total_failed,
        'success_rate': total_successful / total_tasks if total_tasks > 0 else 0,
        'exact_match_rate': exact_matches / total_successful if total_successful > 0 else 0,
        'avg_semantic_similarity': avg_semantic_similarity,
        'avg_evaluation_time': avg_evaluation_time,
        'llm_judge_metrics': {
            'avg_semantic_correctness': avg_llm_semantic_correctness,
            'avg_completeness': avg_llm_completeness,
            'avg_accuracy': avg_llm_accuracy,
            'avg_usefulness': avg_llm_usefulness,
            'avg_overall_score': avg_llm_overall
        },
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
    
    # LLM Judge Metrics
    llm_metrics = summary.get('llm_judge_metrics', {})
    if llm_metrics:
        print("\nLLM Judge Metrics (0-10 scale):")
        print(f"  Semantic Correctness: {llm_metrics['avg_semantic_correctness']:.2f}")
        print(f"  Completeness: {llm_metrics['avg_completeness']:.2f}")
        print(f"  Accuracy: {llm_metrics['avg_accuracy']:.2f}")
        print(f"  Usefulness: {llm_metrics['avg_usefulness']:.2f}")
        print(f"  Overall Score: {llm_metrics['avg_overall_score']:.2f}")
    
    print("\nPer QA Type Performance:")
    for qa_type, stats in summary['qa_type_stats'].items():
        print(f"  {qa_type}: {stats['count']} tasks, "
              f"exact match: {stats['exact_match_rate']:.2%}, "
              f"semantic similarity: {stats['avg_semantic_similarity']:.3f}, "
              f"LLM score: {stats['avg_llm_score']:.2f}")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

