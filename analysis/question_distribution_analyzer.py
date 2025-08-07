#!/usr/bin/env python3

"""
Question Distribution Analyzer

Analyzes the distribution and characteristics of questions and answers
across the DriveLM dataset, including categorization, complexity analysis,
and spatial-temporal relationships.
"""

import re
from typing import Dict, List, Any, Tuple, Union
from collections import Counter, defaultdict
import numpy as np
from analysis.data_loader import DataLoader


class QuestionDistributionAnalyzer:
    """Analyzes question and answer distribution patterns in DriveLM dataset"""
    
    def __init__(self, data_loader: DataLoader):
        """Initialize the analyzer with a data loader"""
        self.data_loader = data_loader
        
        # Define question categories and keywords
        self.question_categories = {
            'object_detection': [
                'detect', 'see', 'visible', 'object', 'car', 'vehicle', 'pedestrian', 
                'bicycle', 'truck', 'bus', 'motorcycle', 'traffic light', 'sign',
                'what is', 'what are', 'identify', 'recognize'
            ],
            'spatial_relationships': [
                'distance', 'position', 'location', 'behind', 'front', 'side', 'left', 'right',
                'near', 'far', 'close', 'between', 'next to', 'opposite', 'relative',
                'spatial', 'geometric', 'where', 'positioned'
            ],
            'temporal_analysis': [
                'time', 'duration', 'speed', 'velocity', 'acceleration', 'when',
                'temporal', 'sequence', 'before', 'after', 'during', 'timing',
                'movement', 'motion', 'trajectory'
            ],
            'safety_assessment': [
                'safe', 'dangerous', 'risk', 'hazard', 'collision', 'crash',
                'emergency', 'warning', 'caution', 'threat', 'safety', 'secure',
                'unsafe', 'risky', 'danger'
            ],
            'traffic_rules': [
                'traffic', 'rule', 'law', 'regulation', 'compliance', 'violation',
                'legal', 'illegal', 'permitted', 'forbidden', 'right of way',
                'priority', 'yield', 'stop', 'go', 'signal', 'light'
            ]
        }
        
        # Define complexity indicators
        self.single_step_indicators = [
            'is', 'are', 'does', 'do', 'can', 'will', 'should', 'would',
            'yes', 'no', 'true', 'false', 'present', 'absent', 'visible', 'invisible'
        ]
        
        self.multi_step_indicators = [
            'why', 'how', 'explain', 'describe', 'analyze', 'compare', 'contrast',
            'because', 'therefore', 'consequently', 'as a result', 'due to',
            'multiple', 'several', 'various', 'different', 'complex'
        ]
        
        # Define answer complexity indicators
        self.simple_answer_indicators = ['yes', 'no', 'true', 'false', 'present', 'absent']
        self.detailed_answer_indicators = ['because', 'since', 'therefore', 'however', 'although', 'while']

    def analyze_qa_distribution(self, scene_id: Union[int, str]) -> Dict[str, Any]:
        """Analyze QA distribution across keyframes in a scene"""
        
        # Get all QA pairs from all keyframes (keyframe_id=0 means all keyframes)
        all_qa_data = self.data_loader.extract_questions_from_keyframe(scene_id, 0)
        
        # Extract all QA pairs
        all_qa_pairs = []
        for keyframe_token, qa_categories in all_qa_data.items():
            if isinstance(qa_categories, dict):
                for category, qa_list in qa_categories.items():
                    if isinstance(qa_list, list):
                        for qa in qa_list:
                            if isinstance(qa, dict) and 'Q' in qa and 'A' in qa:
                                all_qa_pairs.append({
                                    'keyframe_token': keyframe_token,
                                    'category': category,
                                    'question': qa['Q'],
                                    'answer': qa['A']
                                })
        
        # Analyze distribution
        keyframe_counts = Counter([qa['keyframe_token'] for qa in all_qa_pairs])
        category_counts = Counter([qa['category'] for qa in all_qa_pairs])
        
        # Categorize by perception, planning, prediction, behavior
        perception_qa = [qa for qa in all_qa_pairs if self._is_perception_question(qa['question'])]
        planning_qa = [qa for qa in all_qa_pairs if self._is_planning_question(qa['question'])]
        prediction_qa = [qa for qa in all_qa_pairs if self._is_prediction_question(qa['question'])]
        behavior_qa = [qa for qa in all_qa_pairs if self._is_behavior_question(qa['question'])]
        
        return {
            'total_qa_pairs': len(all_qa_pairs),
            'keyframe_distribution': dict(keyframe_counts),
            'category_distribution': dict(category_counts),
            'functional_split': {
                'perception': len(perception_qa),
                'planning': len(planning_qa),
                'prediction': len(prediction_qa),
                'behavior': len(behavior_qa)
            },
            'qa_pairs': all_qa_pairs
        }

    def categorize_questions(self, scene_id: Union[int, str], qa_pairs: List[Dict] = None) -> Dict[str, Any]:
        """Categorize questions into predefined categories"""
        
        if qa_pairs is None:
            qa_distribution = self.analyze_qa_distribution(scene_id)
            qa_pairs = qa_distribution.get('qa_pairs', [])
        
        category_matches = defaultdict(list)
        
        for qa in qa_pairs:
            question = qa['question'].lower()
            matched_categories = []
            
            for category, keywords in self.question_categories.items():
                for keyword in keywords:
                    if keyword.lower() in question:
                        matched_categories.append(category)
                        break
            
            if matched_categories:
                category_matches[matched_categories[0]].append(qa)
            else:
                category_matches['other'].append(qa)
        
        # Calculate category statistics
        category_stats = {}
        for category, qa_list in category_matches.items():
            category_stats[category] = {
                'count': len(qa_list),
                'percentage': len(qa_list) / len(qa_pairs) * 100 if qa_pairs else 0,
                'questions': [qa['question'] for qa in qa_list]
            }
        
        return {
            'total_questions': len(qa_pairs),
            'category_distribution': category_stats,
            'most_common_category': max(category_stats.items(), key=lambda x: x[1]['count'])[0] if category_stats else None
        }

    def analyze_question_complexity(self, scene_id: Union[int, str], qa_pairs: List[Dict] = None) -> Dict[str, Any]:
        """Analyze question complexity (single-step vs multi-step)"""
        
        if qa_pairs is None:
            qa_distribution = self.analyze_qa_distribution(scene_id)
            qa_pairs = qa_distribution.get('qa_pairs', [])
        
        single_step_count = 0
        multi_step_count = 0
        complexity_scores = []
        
        for qa in qa_pairs:
            question = qa['question'].lower()
            
            # Count indicators
            single_indicators = sum(1 for indicator in self.single_step_indicators if indicator in question)
            multi_indicators = sum(1 for indicator in self.multi_step_indicators if indicator in question)
            
            # Determine complexity
            if multi_indicators > single_indicators:
                multi_step_count += 1
                complexity_scores.append(1.0)  # High complexity
            else:
                single_step_count += 1
                complexity_scores.append(0.0)  # Low complexity
        
        return {
            'total_questions': len(qa_pairs),
            'single_step_count': single_step_count,
            'multi_step_count': multi_step_count,
            'single_step_percentage': single_step_count / len(qa_pairs) * 100 if qa_pairs else 0,
            'multi_step_percentage': multi_step_count / len(qa_pairs) * 100 if qa_pairs else 0,
            'average_complexity_score': np.mean(complexity_scores) if complexity_scores else 0,
            'complexity_distribution': {
                'simple': single_step_count,
                'complex': multi_step_count
            }
        }

    def analyze_answer_complexity(self, scene_id: Union[int, str], qa_pairs: List[Dict] = None) -> Dict[str, Any]:
        """Analyze answer complexity (Yes/no vs detailed explanations)"""
        
        if qa_pairs is None:
            qa_distribution = self.analyze_qa_distribution(scene_id)
            qa_pairs = qa_distribution.get('qa_pairs', [])
        
        simple_answers = 0
        detailed_answers = 0
        answer_lengths = []
        
        for qa in qa_pairs:
            answer = qa['answer'].lower()
            answer_length = len(answer.split())
            
            answer_lengths.append(answer_length)
            
            # Check for simple vs detailed indicators
            simple_indicators = sum(1 for indicator in self.simple_answer_indicators if indicator in answer)
            detailed_indicators = sum(1 for indicator in self.detailed_answer_indicators if indicator in answer)
            
            # Classify based on length and indicators
            if answer_length <= 5 or simple_indicators > detailed_indicators:
                simple_answers += 1
            else:
                detailed_answers += 1
        
        return {
            'total_answers': len(qa_pairs),
            'simple_answers': simple_answers,
            'detailed_answers': detailed_answers,
            'simple_answer_percentage': simple_answers / len(qa_pairs) * 100 if qa_pairs else 0,
            'detailed_answer_percentage': detailed_answers / len(qa_pairs) * 100 if qa_pairs else 0,
            'average_answer_length': np.mean(answer_lengths) if answer_lengths else 0,
            'answer_length_distribution': {
                'short': len([l for l in answer_lengths if l <= 5]),
                'medium': len([l for l in answer_lengths if 5 < l <= 20]),
                'long': len([l for l in answer_lengths if l > 20])
            }
        }

    def analyze_object_frequency(self, scene_id: Union[int, str], qa_pairs: List[Dict] = None) -> Dict[str, Any]:
        """Analyze which objects are most frequently asked about"""
        
        if qa_pairs is None:
            qa_distribution = self.analyze_qa_distribution(scene_id)
            qa_pairs = qa_distribution.get('qa_pairs', [])
        
        # Common objects in driving scenarios
        object_keywords = {
            'vehicle': ['car', 'vehicle', 'automobile', 'truck', 'bus', 'van', 'motorcycle'],
            'pedestrian': ['pedestrian', 'person', 'people', 'walker'],
            'bicycle': ['bicycle', 'bike', 'cyclist'],
            'traffic_light': ['traffic light', 'signal', 'light'],
            'traffic_sign': ['sign', 'road sign', 'traffic sign'],
            'road': ['road', 'street', 'lane', 'highway'],
            'intersection': ['intersection', 'crossing', 'junction'],
            'building': ['building', 'house', 'structure'],
            'obstacle': ['obstacle', 'barrier', 'blockage']
        }
        
        object_counts = defaultdict(int)
        object_questions = defaultdict(list)
        
        for qa in qa_pairs:
            question = qa['question'].lower()
            
            for object_type, keywords in object_keywords.items():
                for keyword in keywords:
                    if keyword in question:
                        object_counts[object_type] += 1
                        object_questions[object_type].append(qa['question'])
                        break
        
        # Sort by frequency
        sorted_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_questions': len(qa_pairs),
            'object_frequency': dict(sorted_objects),
            'most_frequent_object': sorted_objects[0][0] if sorted_objects else None,
            'object_questions': dict(object_questions),
            'object_percentages': {
                obj: count / len(qa_pairs) * 100 if qa_pairs else 0
                for obj, count in object_counts.items()
            }
        }

    def analyze_spatial_relationships(self, scene_id: Union[int, str], qa_pairs: List[Dict] = None) -> Dict[str, Any]:
        """Analyze spatial relationship questions"""
        
        if qa_pairs is None:
            qa_distribution = self.analyze_qa_distribution(scene_id)
            qa_pairs = qa_distribution.get('qa_pairs', [])
        
        spatial_keywords = {
            'distance': ['distance', 'far', 'near', 'close', 'away'],
            'position': ['position', 'location', 'where', 'placed'],
            'relative_motion': ['moving', 'approaching', 'receding', 'overtaking'],
            'spatial_relation': ['behind', 'front', 'side', 'left', 'right', 'between'],
            'orientation': ['facing', 'oriented', 'direction', 'heading']
        }
        
        spatial_counts = defaultdict(int)
        spatial_questions = defaultdict(list)
        
        for qa in qa_pairs:
            question = qa['question'].lower()
            
            for spatial_type, keywords in spatial_keywords.items():
                for keyword in keywords:
                    if keyword in question:
                        spatial_counts[spatial_type] += 1
                        spatial_questions[spatial_type].append(qa['question'])
                        break
        
        return {
            'total_spatial_questions': sum(spatial_counts.values()),
            'spatial_distribution': dict(spatial_counts),
            'spatial_percentages': {
                spatial_type: count / len(qa_pairs) * 100 if qa_pairs else 0
                for spatial_type, count in spatial_counts.items()
            },
            'spatial_questions': dict(spatial_questions),
            'most_common_spatial_type': max(spatial_counts.items(), key=lambda x: x[1])[0] if spatial_counts else None
        }

    def analyze_image_correlation(self, scene_id: Union[int, str]) -> Dict[str, Any]:
        """Analyze correlation between QA and images"""
        
        scene_data = self.data_loader.load_scene_data(scene_id)
        if not scene_data:
            return {}
        
        keyframes = scene_data.get('key_frames', {})
        image_paths = keyframes.get('image_paths', {})
        
        # Get all QA pairs
        all_qa_data = self.data_loader.extract_questions_from_keyframe(scene_id, 0)
        
        # Count QA pairs with corresponding images
        qa_with_images = 0
        qa_without_images = 0
        image_qa_pairs = []
        
        for keyframe_token, qa_categories in all_qa_data.items():
            if isinstance(qa_categories, dict):
                # Check if this keyframe has corresponding images
                has_images = keyframe_token in image_paths and image_paths[keyframe_token]
                
                for category, qa_list in qa_categories.items():
                    if isinstance(qa_list, list):
                        for qa in qa_list:
                            if isinstance(qa, dict) and 'Q' in qa and 'A' in qa:
                                if has_images:
                                    qa_with_images += 1
                                    image_qa_pairs.append({
                                        'keyframe_token': keyframe_token,
                                        'category': category,
                                        'question': qa['Q'],
                                        'answer': qa['A'],
                                        'has_image': True
                                    })
                                else:
                                    qa_without_images += 1
                                    image_qa_pairs.append({
                                        'keyframe_token': keyframe_token,
                                        'category': category,
                                        'question': qa['Q'],
                                        'answer': qa['A'],
                                        'has_image': False
                                    })
        
        total_qa = qa_with_images + qa_without_images
        
        return {
            'total_qa_pairs': total_qa,
            'qa_with_images': qa_with_images,
            'qa_without_images': qa_without_images,
            'image_correlation_percentage': qa_with_images / total_qa * 100 if total_qa > 0 else 0,
            'image_qa_pairs': image_qa_pairs,
            'keyframes_with_images': [kf for kf in image_paths.keys() if image_paths[kf]],
            'keyframes_without_images': [kf for kf in all_qa_data.keys() if kf not in image_paths or not image_paths[kf]]
        }

    def generate_comprehensive_qa_analysis(self, scene_id: Union[int, str]) -> Dict[str, Any]:
        """Generate comprehensive QA analysis for a scene"""
        
        # Get QA distribution once to avoid multiple data loader calls
        qa_distribution = self.analyze_qa_distribution(scene_id)
        qa_pairs = qa_distribution.get('qa_pairs', [])
        
        return {
            'qa_distribution': qa_distribution,
            'question_categories': self.categorize_questions(scene_id, qa_pairs),
            'question_complexity': self.analyze_question_complexity(scene_id, qa_pairs),
            'answer_complexity': self.analyze_answer_complexity(scene_id, qa_pairs),
            'object_frequency': self.analyze_object_frequency(scene_id, qa_pairs),
            'spatial_relationships': self.analyze_spatial_relationships(scene_id, qa_pairs),
            'image_correlation': self.analyze_image_correlation(scene_id)
        }

    def _is_perception_question(self, question: str) -> bool:
        """Check if question is perception-related"""
        perception_keywords = ['see', 'visible', 'detect', 'observe', 'notice', 'appear']
        return any(keyword in question.lower() for keyword in perception_keywords)

    def _is_planning_question(self, question: str) -> bool:
        """Check if question is planning-related"""
        planning_keywords = ['plan', 'route', 'path', 'decision', 'choose', 'strategy']
        return any(keyword in question.lower() for keyword in planning_keywords)

    def _is_prediction_question(self, question: str) -> bool:
        """Check if question is prediction-related"""
        prediction_keywords = ['will', 'predict', 'future', 'next', 'likely', 'probability']
        return any(keyword in question.lower() for keyword in prediction_keywords)

    def _is_behavior_question(self, question: str) -> bool:
        """Check if question is behavior-related"""
        behavior_keywords = ['behave', 'action', 'response', 'react', 'should', 'must']
        return any(keyword in question.lower() for keyword in behavior_keywords) 