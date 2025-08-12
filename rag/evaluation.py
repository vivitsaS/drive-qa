"""
Evaluation Module for RAG Pipeline

Provides comprehensive evaluation metrics for comparing generated answers 
with ground truth answers from the dataset.
"""

import re
from typing import Dict, List, Any
import numpy as np
from loguru import logger


class RAGEvaluator:
    """Evaluator for RAG-generated answers against ground truth"""
    
    def __init__(self):
        """Initialize the evaluator"""
        self.metrics = [
            "exact_match",
            "word_overlap",
            "semantic_similarity",
            "length_ratio",
            "keyword_preservation",
            "answer_type_match"
        ]
    
    def evaluate_answer(self, generated: str, ground_truth: str, qa_type: str = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of generated answer vs ground truth.
        
        Args:
            generated: Generated answer from RAG
            ground_truth: Ground truth answer from dataset
            qa_type: Type of QA pair for specialized evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        evaluation = {
            "generated_answer": generated.strip(),
            "ground_truth": ground_truth.strip(),
            "qa_type": qa_type
        }
        
        # Basic metrics
        evaluation.update(self._basic_metrics(generated, ground_truth))
        
        # Advanced metrics
        evaluation.update(self._advanced_metrics(generated, ground_truth))
        
        # QA-type specific metrics
        if qa_type:
            evaluation.update(self._qa_type_specific_metrics(generated, ground_truth, qa_type))
        
        # Overall score
        evaluation["overall_score"] = self._calculate_overall_score(evaluation)
        
        return evaluation
    
    def _basic_metrics(self, generated: str, ground_truth: str) -> Dict[str, float]:
        """Calculate basic evaluation metrics"""
        gen_clean = generated.strip().lower()
        gt_clean = ground_truth.strip().lower()
        
        # Exact match
        exact_match = float(gen_clean == gt_clean)
        
        # Length ratio
        length_ratio = len(generated) / max(len(ground_truth), 1)
        
        # Word-level metrics
        gen_words = set(self._tokenize(gen_clean))
        gt_words = set(self._tokenize(gt_clean))
        
        if gt_words:
            word_precision = len(gen_words.intersection(gt_words)) / len(gen_words) if gen_words else 0
            word_recall = len(gen_words.intersection(gt_words)) / len(gt_words)
            word_f1 = 2 * word_precision * word_recall / (word_precision + word_recall) if (word_precision + word_recall) > 0 else 0
        else:
            word_precision = word_recall = word_f1 = 0
        
        return {
            "exact_match": exact_match,
            "length_ratio": length_ratio,
            "word_precision": word_precision,
            "word_recall": word_recall,
            "word_f1": word_f1
        }
    
    def _advanced_metrics(self, generated: str, ground_truth: str) -> Dict[str, float]:
        """Calculate advanced evaluation metrics"""
        gen_clean = generated.strip().lower()
        gt_clean = ground_truth.strip().lower()
        
        # Semantic similarity (simple version - can be enhanced with embeddings)
        semantic_similarity = self._simple_semantic_similarity(gen_clean, gt_clean)
        
        # Keyword preservation (important terms should be preserved)
        keyword_preservation = self._keyword_preservation_score(gen_clean, gt_clean)
        
        # Answer type consistency (yes/no, numeric, descriptive)
        answer_type_match = self._answer_type_consistency(generated, ground_truth)
        
        return {
            "semantic_similarity": semantic_similarity,
            "keyword_preservation": keyword_preservation,
            "answer_type_match": answer_type_match
        }
    
    def _qa_type_specific_metrics(self, generated: str, ground_truth: str, qa_type: str) -> Dict[str, float]:
        """QA type specific evaluation metrics"""
        metrics = {}
        
        if qa_type == "perception":
            # Focus on object detection accuracy
            metrics["object_accuracy"] = self._object_detection_accuracy(generated, ground_truth)
            metrics["spatial_accuracy"] = self._spatial_accuracy(generated, ground_truth)
            
        elif qa_type == "planning":
            # Focus on action/decision accuracy
            metrics["action_accuracy"] = self._action_accuracy(generated, ground_truth)
            metrics["reasoning_quality"] = self._reasoning_quality(generated, ground_truth)
            
        elif qa_type == "prediction":
            # Focus on future state accuracy
            metrics["prediction_accuracy"] = self._prediction_accuracy(generated, ground_truth)
            metrics["temporal_accuracy"] = self._temporal_accuracy(generated, ground_truth)
            
        elif qa_type == "behavior":
            # Focus on explanation quality
            metrics["explanation_quality"] = self._explanation_quality(generated, ground_truth)
            metrics["causal_accuracy"] = self._causal_accuracy(generated, ground_truth)
        
        return metrics
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return re.findall(r'\b\w+\b', text.lower())
    
    def _simple_semantic_similarity(self, gen: str, gt: str) -> float:
        """Simple semantic similarity based on word overlap"""
        gen_words = set(self._tokenize(gen))
        gt_words = set(self._tokenize(gt))
        
        all_words = gen_words.union(gt_words)
        if not all_words:
            return 0.0
        
        common_words = gen_words.intersection(gt_words)
        return len(common_words) / len(all_words)
    
    def _keyword_preservation_score(self, gen: str, gt: str) -> float:
        """Score based on preservation of important keywords"""
        # Identify important keywords (nouns, adjectives, specific terms)
        important_patterns = [
            r'\b(?:car|vehicle|truck|bike|pedestrian|object)\b',
            r'\b(?:left|right|front|back|behind|ahead)\b',
            r'\b(?:speed|fast|slow|stop|brake|accelerate)\b',
            r'\b(?:red|green|yellow|blue|white|black)\b',
            r'\b(?:yes|no|true|false)\b'
        ]
        
        gt_keywords = set()
        for pattern in important_patterns:
            gt_keywords.update(re.findall(pattern, gt))
        
        if not gt_keywords:
            return 1.0
        
        gen_keywords = set()
        for pattern in important_patterns:
            gen_keywords.update(re.findall(pattern, gen))
        
        preserved = len(gt_keywords.intersection(gen_keywords))
        return preserved / len(gt_keywords)
    
    def _answer_type_consistency(self, gen: str, gt: str) -> float:
        """Check if answer types are consistent (yes/no, numeric, etc.)"""
        def classify_answer_type(text: str) -> str:
            text = text.strip().lower()
            
            if re.match(r'^(yes|no|true|false)\.?$', text):
                return "boolean"
            elif re.search(r'\d+(?:\.\d+)?', text):
                return "numeric"
            elif len(text.split()) <= 3:
                return "short"
            else:
                return "descriptive"
        
        gen_type = classify_answer_type(gen)
        gt_type = classify_answer_type(gt)
        
        return 1.0 if gen_type == gt_type else 0.5
    
    def _object_detection_accuracy(self, gen: str, gt: str) -> float:
        """Evaluate object detection accuracy for perception questions"""
        object_keywords = ['car', 'vehicle', 'truck', 'bike', 'bicycle', 'pedestrian', 'person', 'bus', 'motorcycle']
        
        gen_objects = set()
        gt_objects = set()
        
        for keyword in object_keywords:
            if keyword in gen.lower():
                gen_objects.add(keyword)
            if keyword in gt.lower():
                gt_objects.add(keyword)
        
        if not gt_objects:
            return 1.0 if not gen_objects else 0.5
        
        correct_objects = len(gen_objects.intersection(gt_objects))
        return correct_objects / len(gt_objects)
    
    def _spatial_accuracy(self, gen: str, gt: str) -> float:
        """Evaluate spatial relationship accuracy"""
        spatial_keywords = ['left', 'right', 'front', 'back', 'behind', 'ahead', 'beside', 'near', 'far']
        
        gen_spatial = set()
        gt_spatial = set()
        
        for keyword in spatial_keywords:
            if keyword in gen.lower():
                gen_spatial.add(keyword)
            if keyword in gt.lower():
                gt_spatial.add(keyword)
        
        if not gt_spatial:
            return 1.0
        
        correct_spatial = len(gen_spatial.intersection(gt_spatial))
        return correct_spatial / len(gt_spatial)
    
    def _action_accuracy(self, gen: str, gt: str) -> float:
        """Evaluate action/decision accuracy for planning questions"""
        action_keywords = ['stop', 'go', 'turn', 'brake', 'accelerate', 'slow', 'yield', 'wait', 'proceed']
        
        gen_actions = set()
        gt_actions = set()
        
        for keyword in action_keywords:
            if keyword in gen.lower():
                gen_actions.add(keyword)
            if keyword in gt.lower():
                gt_actions.add(keyword)
        
        if not gt_actions:
            return 1.0
        
        correct_actions = len(gen_actions.intersection(gt_actions))
        return correct_actions / len(gt_actions)
    
    def _reasoning_quality(self, gen: str, gt: str) -> float:
        """Evaluate reasoning quality"""
        reasoning_indicators = ['because', 'since', 'due to', 'as', 'therefore', 'so', 'thus']
        
        gen_has_reasoning = any(indicator in gen.lower() for indicator in reasoning_indicators)
        gt_has_reasoning = any(indicator in gt.lower() for indicator in reasoning_indicators)
        
        if gt_has_reasoning:
            return 1.0 if gen_has_reasoning else 0.5
        else:
            return 1.0
    
    def _prediction_accuracy(self, gen: str, gt: str) -> float:
        """Evaluate prediction accuracy"""
        prediction_keywords = ['will', 'going to', 'likely', 'probably', 'might', 'could', 'would']
        
        gen_has_prediction = any(keyword in gen.lower() for keyword in prediction_keywords)
        gt_has_prediction = any(keyword in gt.lower() for keyword in prediction_keywords)
        
        if gt_has_prediction:
            return 1.0 if gen_has_prediction else 0.5
        else:
            return 1.0
    
    def _temporal_accuracy(self, gen: str, gt: str) -> float:
        """Evaluate temporal accuracy"""
        temporal_keywords = ['now', 'soon', 'later', 'next', 'after', 'before', 'currently', 'future']
        
        gen_temporal = set()
        gt_temporal = set()
        
        for keyword in temporal_keywords:
            if keyword in gen.lower():
                gen_temporal.add(keyword)
            if keyword in gt.lower():
                gt_temporal.add(keyword)
        
        if not gt_temporal:
            return 1.0
        
        correct_temporal = len(gen_temporal.intersection(gt_temporal))
        return correct_temporal / len(gt_temporal)
    
    def _explanation_quality(self, gen: str, gt: str) -> float:
        """Evaluate explanation quality for behavior questions"""
        # Simple heuristic: longer explanations with reasoning words are better
        explanation_words = ['because', 'reason', 'cause', 'explain', 'due', 'result', 'purpose']
        
        gen_score = sum(1 for word in explanation_words if word in gen.lower())
        gt_score = sum(1 for word in explanation_words if word in gt.lower())
        
        if gt_score == 0:
            return 1.0
        
        return min(gen_score / gt_score, 1.0)
    
    def _causal_accuracy(self, gen: str, gt: str) -> float:
        """Evaluate causal relationship accuracy"""
        causal_keywords = ['cause', 'effect', 'result', 'lead', 'trigger', 'because', 'due to']
        
        gen_causal = any(keyword in gen.lower() for keyword in causal_keywords)
        gt_causal = any(keyword in gt.lower() for keyword in causal_keywords)
        
        if gt_causal:
            return 1.0 if gen_causal else 0.5
        else:
            return 1.0
    
    def _calculate_overall_score(self, evaluation: Dict[str, Any]) -> float:
        """Calculate overall score from individual metrics"""
        # Weight different metrics based on importance
        weights = {
            "exact_match": 0.2,
            "word_f1": 0.3,
            "semantic_similarity": 0.25,
            "keyword_preservation": 0.15,
            "answer_type_match": 0.1
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in evaluation:
                score += evaluation[metric] * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def batch_evaluate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate a batch of results and provide aggregated metrics.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Aggregated evaluation metrics
        """
        if not results:
            return {"error": "No results to evaluate"}
        
        # Aggregate metrics
        aggregated = {
            "total_pairs": len(results),
            "metrics": {}
        }
        
        # Collect all metrics
        for metric in self.metrics:
            values = [r.get(metric, 0) for r in results if metric in r]
            if values:
                aggregated["metrics"][metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
        
        # QA type breakdown
        qa_types = {}
        for result in results:
            qa_type = result.get("qa_type")
            if qa_type:
                if qa_type not in qa_types:
                    qa_types[qa_type] = []
                qa_types[qa_type].append(result)
        
        aggregated["by_qa_type"] = {}
        for qa_type, type_results in qa_types.items():
            type_metrics = {}
            for metric in self.metrics:
                values = [r.get(metric, 0) for r in type_results if metric in r]
                if values:
                    type_metrics[metric] = np.mean(values)
            aggregated["by_qa_type"][qa_type] = type_metrics
        
        return aggregated