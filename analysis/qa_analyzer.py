"""
QA Analyzer for DriveLM Dataset

Analyzes question-answer pairs across different levels:
- Overall analysis (all scenes)
- Scene-level analysis
- Keyframe-level analysis
"""

from collections import Counter, defaultdict
import json
import re
from typing import Any, Dict, List, Tuple, Union

from loguru import logger
import numpy as np

from .data_loader import DataLoader


class QAAnalyzer:
    """Analyze QA pairs from DriveLM dataset"""
    
    def __init__(self, data_loader: DataLoader = None, scene_id: Union[int, str] = None):
        """
        Initialize QA analyzer.
        
        Args:
            data_loader: DataLoader instance, creates new one if None
        """
        self.data_loader = data_loader if data_loader else DataLoader()
        self.qa_types = ['perception', 'planning', 'prediction', 'behavior']
        self.scene_id = scene_id
    
    def analyze_scenes(self) -> Dict[str, Any]:
        """
        get all the qa_distribution for all keyframes in a scene.
        """
        total_dict = {}
        for scene_id in range(1, 7):
            x = self._get_qa_distribution(scene_id, 0)
            logger.info(f"QA distribution for scene {scene_id}: {x}")
            total_dict[scene_id] = x
        total_all_scenes = {}
        # now we need to get the total for each qa_type.
        total_all_scenes["total"] = sum(total_dict[scene_id]["total"] for scene_id in total_dict)
        total_all_scenes["perception"] = sum(total_dict[scene_id]["perception"] for scene_id in total_dict)
        total_all_scenes["planning"] = sum(total_dict[scene_id]["planning"] for scene_id in total_dict)
        total_all_scenes["prediction"] = sum(total_dict[scene_id]["prediction"] for scene_id in total_dict)
        total_all_scenes["behavior"] = sum(total_dict[scene_id]["behavior"] for scene_id in total_dict)
        logger.info(f"Total QA distribution: {total_all_scenes}")
        return total_all_scenes
    
    def analyze_qa_content(self) -> Dict[str, Any]:
        """
        Analyze content patterns in QA data across all scenes.
        
        Returns:
            Dictionary containing object mentions, question patterns, and answer characteristics
        """
        logger.info("Analyzing QA content patterns...")
        
        # Get all QA data for analysis
        all_qa_data = {}
        for scene_id in range(1, 7):
            scene_data = self.data_loader.load_scene_data(scene_id)
            for keyframe_token in scene_data['key_frames']:
                qa_data = scene_data['key_frames'][keyframe_token]['QA']
                all_qa_data[f"scene_{scene_id}_{keyframe_token}"] = qa_data
        
        # Analyze content patterns
        object_mentions = self._extract_all_object_mentions(all_qa_data)
        object_mentions_by_type = self._extract_object_mentions_by_qa_type(all_qa_data)
        question_patterns = self._analyze_question_patterns(all_qa_data)
        answer_patterns = self._analyze_answer_patterns(all_qa_data)
        answer_characteristics = self._analyze_answer_characteristics(all_qa_data)
        
        return {
            'objects': object_mentions,
            'objects_by_type': object_mentions_by_type,
            'question_patterns': question_patterns,
            'answer_patterns': answer_patterns,
            'answer_characteristics': answer_characteristics
        }
    
    def _extract_all_object_mentions(self, all_qa_data: Dict[str, Any]) -> Dict[str, int]:
        """Extract object mentions from all QA data"""
        object_mentions = Counter()
        
        # Common object patterns
        object_patterns = [
            r'\b(car|cars|vehicle|vehicles)\b',
            r'\b(pedestrian|pedestrians|person|people)\b',
            r'\b(bicycle|bicycles|bike|bikes)\b',
            r'\b(motorcycle|motorcycles)\b',
            r'\b(truck|trucks)\b',
            r'\b(bus|buses)\b',
            r'\b(traffic light|traffic lights)\b',
            r'\b(stop sign|stop signs)\b',
            r'\b(barrier|barriers)\b',
            r'\b(traffic cone|traffic cones)\b',
            r'\b(construction|construction vehicle)\b'
        ]
        
        for scene_keyframe, qa_data in all_qa_data.items():
            for qa_type in self.qa_types:
                if qa_type in qa_data:
                    for qa_pair in qa_data[qa_type]:
                        question = qa_pair.get('Q', '').lower()
                        answer = qa_pair.get('A', '').lower()
                        
                        for pattern in object_patterns:
                            matches = re.findall(pattern, question + ' ' + answer)
                            for match in matches:
                                object_mentions[match] += 1
        
        return dict(object_mentions.most_common(15))  # Top 15 objects
    
    def _extract_object_mentions_by_qa_type(self, all_qa_data: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
        """Extract object mentions broken down by QA type"""
        object_mentions_by_type = {qa_type: Counter() for qa_type in self.qa_types}
        
        # Common object patterns
        object_patterns = [
            r'\b(car|cars|vehicle|vehicles)\b',
            r'\b(pedestrian|pedestrians|person|people)\b',
            r'\b(bicycle|bicycles|bike|bikes)\b',
            r'\b(motorcycle|motorcycles)\b',
            r'\b(truck|trucks)\b',
            r'\b(bus|buses)\b',
            r'\b(traffic light|traffic lights)\b',
            r'\b(stop sign|stop signs)\b',
            r'\b(barrier|barriers)\b',
            r'\b(traffic cone|traffic cones)\b',
            r'\b(construction|construction vehicle)\b'
        ]
        
        for scene_keyframe, qa_data in all_qa_data.items():
            for qa_type in self.qa_types:
                if qa_type in qa_data:
                    for qa_pair in qa_data[qa_type]:
                        question = qa_pair.get('Q', '').lower()
                        answer = qa_pair.get('A', '').lower()
                        
                        for pattern in object_patterns:
                            matches = re.findall(pattern, question + ' ' + answer)
                            for match in matches:
                                object_mentions_by_type[qa_type][match] += 1
        
        # Convert to regular dict and get top objects
        result = {}
        for qa_type in self.qa_types:
            result[qa_type] = dict(object_mentions_by_type[qa_type].most_common(10))  # Top 10 per type
        
        return result
    
    def _analyze_question_patterns(self, all_qa_data: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
        """Analyze question patterns by QA type"""
        question_patterns = defaultdict(lambda: defaultdict(int))
        
        # Question pattern keywords
        question_patterns_keywords = {
            'what': ['what', 'what is', 'what are'],
            'where': ['where', 'where is', 'where are'],
            'when': ['when', 'when will', 'when should'],
            'how': ['how', 'how should', 'how will'],
            'why': ['why', 'why should', 'why will'],
            'status': ['status', 'state', 'condition'],
            'action': ['should', 'will', 'going to', 'planning to']
        }
        
        for scene_keyframe, qa_data in all_qa_data.items():
            for qa_type in self.qa_types:
                if qa_type in qa_data:
                    for qa_pair in qa_data[qa_type]:
                        question = qa_pair.get('Q', '').lower()
                        
                        for pattern_name, keywords in question_patterns_keywords.items():
                            for keyword in keywords:
                                if keyword in question:
                                    question_patterns[pattern_name][qa_type] += 1
                                    break
        
        return dict(question_patterns)
    
    def _analyze_answer_patterns(self, all_qa_data: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
        """Analyze answer patterns by QA type"""
        answer_patterns = defaultdict(lambda: defaultdict(int))
        
        # Answer pattern keywords
        answer_patterns_keywords = {
            'descriptive': ['there are', 'there is', 'many', 'several', 'various'],
            'actionable': ['should', 'will', 'need to', 'must', 'have to'],
            'conditional': ['if', 'when', 'unless', 'provided that', 'in case'],
            'temporal': ['now', 'soon', 'later', 'before', 'after', 'while'],
            'spatial': ['front', 'back', 'left', 'right', 'near', 'far', 'behind'],
            'quantitative': ['one', 'two', 'three', 'many', 'few', 'several', 'all'],
            'qualitative': ['good', 'bad', 'safe', 'dangerous', 'clear', 'obstructed']
        }
        
        for scene_keyframe, qa_data in all_qa_data.items():
            for qa_type in self.qa_types:
                if qa_type in qa_data:
                    for qa_pair in qa_data[qa_type]:
                        answer = qa_pair.get('A', '').lower()
                        
                        for pattern_name, keywords in answer_patterns_keywords.items():
                            for keyword in keywords:
                                if keyword in answer:
                                    answer_patterns[pattern_name][qa_type] += 1
                                    break
        
        return dict(answer_patterns)
    
    def _analyze_answer_characteristics(self, all_qa_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze answer characteristics by QA type"""
        answer_lengths = defaultdict(list)
        answer_complexity = defaultdict(list)
        
        for scene_keyframe, qa_data in all_qa_data.items():
            for qa_type in self.qa_types:
                if qa_type in qa_data:
                    for qa_pair in qa_data[qa_type]:
                        answer = qa_pair.get('A', '').lower()
                        
                        # Answer length (word count)
                        word_count = len(answer.split())
                        answer_lengths[qa_type].append(word_count)
                        
                        # Answer complexity (sentence count)
                        sentence_count = len([s for s in answer.split('.') if s.strip()])
                        answer_complexity[qa_type].append(sentence_count)
        
        return {
            'lengths': dict(answer_lengths),
            'complexity': dict(answer_complexity)
        }
        
    def _get_qa_distribution(self, scene_id: Union[int, str], keyframe_id: Union[int, str]) -> Dict[str, Any]:
        """
        returns a dict {"perception": <total perception qa_pairs>, "planning": <total planning qa_pairs>, "prediction": <total prediction qa_pairs>, "behavior": <total behavior qa_pairs>, "total": <total qa_pairs>}
        """
        qa_data = self.data_loader.extract_questions_from_keyframe(scene_id, keyframe_id)
        qa_distribution = {qa_type: 0 for qa_type in self.qa_types}        
        qa_distribution["total"] = 0
        if keyframe_id != 0:

            for qa_type in self.qa_types:
                qa_distribution[qa_type] = len(qa_data[qa_type])
                qa_distribution["total"] += len(qa_data[qa_type])
            # logger.info(f"QA distribution: {qa_distribution}")
            return qa_distribution
        # recursive call for all keyframes. here the qa_data looks like this: {"<keyframe_token>": {"perception": [qa_pairs], "planning": [qa_pairs], "prediction": [qa_pairs], "behavior": [qa_pairs]}}

        for keyframe_token, qa_data in qa_data.items():
            qa_distribution_for_keyframe = self._get_qa_distribution(scene_id, keyframe_token)
            for qa_type in self.qa_types:
                qa_distribution[qa_type] += qa_distribution_for_keyframe[qa_type]
            # Recalculate total from individual QA types to avoid double counting
            qa_distribution["total"] = sum(qa_distribution[qa_type] for qa_type in self.qa_types)
            # logger.info(f"QA distribution total: {qa_distribution}")
        return qa_distribution
  
    def _extract_object_mentions(self, qa_data: Dict[str, List[Dict]]) -> Dict[str, int]:
        """Extract object mentions from QA data"""
        object_mentions = defaultdict(int)
        
        for qa_list in qa_data.values():
            for qa_pair in qa_list:
                question = qa_pair.get('Q', '').lower()
                answer = qa_pair.get('A', '').lower()
                
                objects = self._extract_objects_from_text(question + ' ' + answer)
                for obj in objects:
                    object_mentions[obj] += 1
        
        return dict(object_mentions)
    
    def _extract_objects_from_text(self, text: str) -> List[str]:
        """Extract object names from text"""
        # Common object categories in driving scenarios
        object_patterns = [
            r'\b(car|cars|vehicle|vehicles)\b',
            r'\b(pedestrian|pedestrians|person|people)\b',
            r'\b(bicycle|bicycles|bike|bikes)\b',
            r'\b(motorcycle|motorcycles)\b',
            r'\b(truck|trucks)\b',
            r'\b(bus|buses)\b',
            r'\b(traffic light|traffic lights)\b',
            r'\b(stop sign|stop signs)\b',
            r'\b(barrier|barriers)\b',
            r'\b(traffic cone|traffic cones)\b',
            r'\b(construction|construction vehicle)\b'
        ]
        
        objects = []
        for pattern in object_patterns:
            matches = re.findall(pattern, text)
            objects.extend(matches)
        
        return list(set(objects))
    
    def _extract_scenario_indicators(self, qa_data: Dict[str, List[Dict]]) -> Dict[str, int]:
        """Extract scenario indicators from QA data"""
        scenario_indicators = defaultdict(int)
        
        scenario_patterns = {
            'turning': [r'\b(turn|turning|left|right)\b'],
            'stopping': [r'\b(stop|stopping|halt|wait)\b'],
            'crossing': [r'\b(cross|crossing|intersection)\b'],
            'parking': [r'\b(park|parking|parked)\b'],
            'overtaking': [r'\b(overtake|passing|pass)\b'],
            'lane_change': [r'\b(lane|change|merge)\b'],
            'braking': [r'\b(brake|braking|stop)\b']
        }
        
        for qa_list in qa_data.values():
            for qa_pair in qa_list:
                question = qa_pair.get('Q', '').lower()
                answer = qa_pair.get('A', '').lower()
                text = question + ' ' + answer
                
                for scenario, patterns in scenario_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, text):
                            scenario_indicators[scenario] += 1
        
        return dict(scenario_indicators)
    
    def _extract_risk_indicators(self, qa_data: Dict[str, List[Dict]]) -> Dict[str, int]:
        """Extract risk indicators from QA data"""
        risk_indicators = defaultdict(int)
        
        risk_patterns = {
            'high_risk': [r'\b(dangerous|risky|hazard|emergency)\b'],
            'collision': [r'\b(collision|crash|hit|impact)\b'],
            'near_miss': [r'\b(near|close|almost|narrowly)\b'],
            'speed': [r'\b(fast|speed|accelerate|slow)\b'],
            'visibility': [r'\b(visibility|visible|hidden|obscured)\b']
        }
        
        for qa_list in qa_data.values():
            for qa_pair in qa_list:
                question = qa_pair.get('Q', '').lower()
                answer = qa_pair.get('A', '').lower()
                text = question + ' ' + answer
                
                for risk_type, patterns in risk_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, text):
                            risk_indicators[risk_type] += 1
        
        return dict(risk_indicators)
    
    def _calculate_object_importance(self, object_frequency: Dict[str, Dict[str, int]]) -> Dict[str, float]:
        """Calculate importance ranking for objects based on QA frequency"""
        object_scores = {}
        
        for obj, qa_counts in object_frequency.items():
            # Calculate importance based on frequency across all QA types
            total_mentions = sum(qa_counts.values())
            qa_type_diversity = len([count for count in qa_counts.values() if count > 0])
            
            # Score based on total mentions and diversity across QA types
            importance_score = total_mentions * qa_type_diversity
            object_scores[obj] = importance_score
        
        # Sort by importance score
        sorted_objects = sorted(object_scores.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_objects)
    
    def _classify_scenario(self, scene_description: str) -> str:

        """Classify driving scenario based on scene description"""
        description_lower = scene_description.lower()
        
        if any(word in description_lower for word in ['intersection', 'crossing']):
            return 'intersection'
        elif any(word in description_lower for word in ['highway', 'freeway']):
            return 'highway'
        elif any(word in description_lower for word in ['parking', 'parked']):
            return 'parking'
        elif any(word in description_lower for word in ['construction', 'barrier']):
            return 'construction'
        elif any(word in description_lower for word in ['night', 'dark']):
            return 'night_driving'
        elif any(word in description_lower for word in ['pedestrian', 'crossing']):
            return 'pedestrian_heavy'
        else:
            return 'urban_general' 

    def analyze_keyframe(self, scene_id: Union[int, str], keyframe_id: Union[int, str]) -> Dict[str, Any]:
        """
        Analyze QA patterns for a specific keyframe.
        
        Args:
            scene_id: Scene identifier (int or str)
            keyframe_id: Keyframe identifier (int or str)
            
        Returns:
            Dictionary with keyframe QA analysis
        """
        # logger.info(f"Analyzing QA patterns for keyframe {keyframe_id} from scene {scene_id} ")
        qa_data = self.data_loader.extract_questions_from_keyframe(scene_id, keyframe_id)
        # logger.info(f"QA data: {qa_data.keys()}")

        keyframe_analysis = {
            # 'scene': {"scene_id": scene_id, "scene_token": self.data_loader._assign_scene_token(scene_id)},
            # 'keyframe': {"keyframe_id": keyframe_id, "keyframe_token": self.data_loader._assign_keyframe_token(scene_id, keyframe_id)},
            'qa_type_distribution': self._get_qa_distribution(scene_id, keyframe_id),
            # 'qa_complexity': self._calculate_qa_complexity(qa_data),
            'object_mentions': self._extract_object_mentions(qa_data),
            'scenario_indicators': self._extract_scenario_indicators(qa_data),
            'risk_indicators': self._extract_risk_indicators(qa_data)
        }        
        return keyframe_analysis


    # def analyze_all_keyframes(self, scene_id: Union[int, str]) -> Dict[str, Any]:
    #     """
    #     Analyze QA patterns for a specific scene.
        
    #     Args:
    #         scene_id: Scene identifier (int or str)
            
    #     Returns:
    #         Dictionary with scene QA analysis
    #     """
    #     logger.info(f"Analyzing QA patterns for scene {self.scene_id}")
    #     scene_data = self.data_loader.load_scene_data(self.scene_id)
    #     qa_data = self.data_loader.extract_questions_from_keyframe(scene_id, 0)

    #     all_keyframe_analysis = {}
    #     for keyframe_id in scene_data['key_frames']:
    #         keyframe_analysis = self.analyze_keyframe_qa(scene_id, keyframe_id)
    #         all_keyframe_analysis[keyframe_id] = keyframe_analysis
    #     logger.info(f"All keyframe analysis: {all_keyframe_analysis}")
    #     # scene_analysis = {
    #     #     'total_keyframes': len(scene_data['key_frames']),
    #     #     'qa_type_distribution': self._get_qa_distribution(scene_id, 0),
    #     #     # 'object_mentions_total': all_keyframe_analysis['object_mentions'],
    #     #     # risk_indicators of this scene
    #     #     'risk_indicators': self._extract_risk_indicators(qa_data),
    #     #     # correlation between qa type and object mentions.
    #     #     'correlation_between_qa_type_and_object_mentions': self._calculate_correlation_between_qa_type_and_object_mentions(qa_data),
    #     #     # questions progression across keyframes. which means evaluate how the questions are changing across keyframes.
    #     #     'questions_progression_across_keyframes': self._analyze_keyframe_progression(qa_data),
    #     #     # question complexity- one word answer or multiple word answer/ subjective or objective
    #     #     'question_complexity': self._calculate_question_complexity(qa_data),
    #     # }
    #     return all_keyframe_analysis

