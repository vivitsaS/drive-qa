"""
Data Loader for concatenated Dataset

Loads and parses concatenated JSON data for analysis.
"""

import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from loguru import logger


class DataLoader:
    """Load and parse concatenated JSON data"""
    
    def __init__(self, data_path: str = "concatenated_data/concatenated_data.json"):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the concatenated JSON data file
        """
        self.data_path = self._assign_data_path(data_path)
    def _assign_data_path(self, data_path: str) -> str:
        """Assign data path"""
        try:
            if data_path is None:
                return "data/concatenated_data/concatenated_data.json"
            else:
                # check if data_path is a valid path
                if not Path(data_path).exists():
                    logger.error(f"Data path {data_path} does not exist")
                    return "data/concatenated_data/concatenated_data.json"
                else:
                    return data_path
        except Exception as e:
            logger.error(f"Error assigning data path: {e}")
            return "data/concatenated_data/concatenated_data.json"
    def _assign_scene_token(self, scene_id) -> str:
        """
        Assign scene id to the scene token.
        """
        # scene_tokens to serial number map
        # get the list of scene tokens
        # get the entire data, which is of the form { "<scene_token>": <scene_data>, "<scene_token>": <scene_data>, ... }
        all_data = self.load_all_data()
        scene_tokens = list(all_data.keys())
        scene_serial_numbers = list(range(1, len(scene_tokens) + 1))
        serial_number_to_scene_token_map = dict(zip(scene_serial_numbers, scene_tokens))
        try:
            if isinstance(scene_id, int):
                if scene_id in serial_number_to_scene_token_map:
                    return serial_number_to_scene_token_map[scene_id]
                else:
                    raise ValueError(f"Scene ID {scene_id} not found in the data, make sure the number is between 1 and {len(scene_tokens)}")
            elif isinstance(scene_id, str):
                if scene_id in scene_tokens:
                    return scene_id
                else:
                    raise ValueError(f"Scene ID {scene_id} not found in the data, make sure the scene token is valid")
        except Exception as e:
            logger.error(f"Error assigning scene token: {e}")
            raise ValueError(f"Invalid scene_id: {scene_id}")
    
    def load_scene_data(self, scene_identifier: str) -> Dict[str, Any]:
        """
        Load scene data from JSON by scene token or serial number.
        
        Args:
            scene_identifier: Scene token or serial number (1-6)
            
        Returns:
            Scene data dictionary
        """
        scene_token = self._assign_scene_token(scene_identifier)
        scene_data = self.load_all_data()[scene_token]
        return scene_data
    
    def load_all_data(self) -> Dict[str, Any]:
        """
        Load all available scene data.
        
        Returns:
            Dictionary containing scene data with scene tokens as keys
        """
        # load the data from the json file
        try:
            with open(self.data_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return {}
        return data
    
    def _assign_keyframe_token(self, scene_token: str, keyframe_id: Union[int, str]) -> str:
        """
        Assign keyframe token to the keyframe id.
        """
        # keyframe_id to keyframe_token map
        try:
            keyframe_tokens = list(self.load_scene_data(scene_token)['key_frames'].keys())
            keyframe_serial_numbers = list(range(1, len(keyframe_tokens) + 1))  
            serial_number_to_keyframe_token_map = dict(zip(keyframe_serial_numbers, keyframe_tokens))
            
            if isinstance(keyframe_id, int):
                if keyframe_id in serial_number_to_keyframe_token_map:
                    return serial_number_to_keyframe_token_map[keyframe_id]
                else:
                    raise ValueError(f"Keyframe ID {keyframe_id} not found in the data, make sure the number is between 1 and {len(keyframe_tokens)}")
            if isinstance(keyframe_id, str):
                if keyframe_id in keyframe_tokens:
                    return keyframe_id
                else:
                    raise ValueError(f"Keyframe ID {keyframe_id} not found in the data, make sure the keyframe token is valid")
        except Exception as e:
            logger.error(f"Error assigning keyframe token: {e}")
            raise ValueError(f"Invalid keyframe_id: {keyframe_id}")
    
    def extract_questions_from_keyframe(self, scene_id: int, keyframe_id: int) -> List[Dict[str, Any]]:
        """Extract all questions from given keyframe"""
        scene_data = self.load_scene_data(scene_id)
        # base case: if keyframe_id is 0, return all questions
        if keyframe_id != 0:
            logger.info(f"Extracting questions from keyframe {keyframe_id}")
            qa_pairs = self.load_scene_data(scene_id)["key_frames"][self._assign_keyframe_token(scene_id, keyframe_id)]["QA"]
            return qa_pairs
        
        all_qa_pairs = {}
        for i in range(1, len(scene_data["key_frames"]) + 1):
            logger.info(f"Extracting questions from keyframe {i}")
            qa_pairs = self.extract_questions_from_keyframe(scene_id, i)
            all_qa_pairs[self._assign_keyframe_token(scene_id, i)] = qa_pairs
        return all_qa_pairs


    def extract_questions_from_scene(self, scene_id: int) -> List[Dict[str, Any]]:
        """
        Extract all questions from scene.
        
        Args:
            scene_id: Scene ID
            
        Returns:
            List of question dictionaries with metadata
        """
        scene_data = self.load_scene_data(scene_id)
        return self.extract_questions_from_keyframe(scene_id, 0)
    
    def extract_objects(self, scene_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract object detection data from scene keyframes.
        
        Args:
            scene_data: Scene data dictionary
            
        Returns:
            List of object dictionaries with metadata
        """
        objects = []
        
        if 'key_frames' not in scene_data:
            return objects
        
        for frame_token, frame_data in scene_data['key_frames'].items():
            if 'key_object_infos' in frame_data:
                for object_id, object_info in frame_data['key_object_infos'].items():
                    object_data = {
                        'scene_token': scene_data['scene_token'],
                        'scene_name': scene_data['scene_name'],
                        'frame_token': frame_token,
                        'object_id': object_id,
                        'category': object_info.get('Category', ''),
                        'status': object_info.get('Status'),
                        'visual_description': object_info.get('Visual_description', ''),
                        'bbox_2d': object_info.get('2d_bbox', [])
                    }
                    objects.append(object_data)
        
        return objects
    
    def extract_spatial_data(self, scene_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract spatial relationship data from scene.
        
        Args:
            scene_data: Scene data dictionary
            
        Returns:
            List of spatial relationship dictionaries
        """
        spatial_data = []
        
        # Extract spatial information from object positions and relationships
        objects = self.extract_objects(scene_data)
        
        for obj in objects:
            if obj['bbox_2d']:
                spatial_info = {
                    'scene_token': obj['scene_token'],
                    'frame_token': obj['frame_token'],
                    'object_id': obj['object_id'],
                    'category': obj['category'],
                    'position': {
                        'x': obj['bbox_2d'][0] if len(obj['bbox_2d']) >= 1 else None,
                        'y': obj['bbox_2d'][1] if len(obj['bbox_2d']) >= 2 else None,
                        'width': obj['bbox_2d'][2] - obj['bbox_2d'][0] if len(obj['bbox_2d']) >= 3 else None,
                        'height': obj['bbox_2d'][3] - obj['bbox_2d'][1] if len(obj['bbox_2d']) >= 4 else None
                    }
                }
                spatial_data.append(spatial_info)
        
        return spatial_data
    
    def extract_temporal_data(self, scene_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract temporal sequence data from scene.
        
        Args:
            scene_data: Scene data dictionary
            
        Returns:
            List of temporal sequence dictionaries
        """
        temporal_data = []
        
        if 'key_frames' not in scene_data:
            return temporal_data
        
        frame_tokens = list(scene_data['key_frames'].keys())
        
        for i, frame_token in enumerate(frame_tokens):
            frame_data = scene_data['key_frames'][frame_token]
            temporal_info = {
                'scene_token': scene_data['scene_token'],
                'scene_name': scene_data['scene_name'],
                'frame_token': frame_token,
                'frame_index': i,
                'total_frames': len(frame_tokens),
                'has_qa': 'QA' in frame_data,
                'has_objects': 'key_object_infos' in frame_data,
                'qa_types': list(frame_data.get('QA', {}).keys()) if 'QA' in frame_data else []
            }
            temporal_data.append(temporal_info)
        
        return temporal_data
    
    def get_available_files(self) -> List[str]:
        """
        Get list of available scene tokens.
        
        Returns:
            List of scene token strings
        """
        try:
            with open(self.data_path, 'r') as f:
                data = json.load(f)
            
            return list(data.keys())
            
        except FileNotFoundError:
            logger.error(f"Data file not found at {self.data_path}")
            return []
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in {self.data_path}")
            return []
        except Exception as e:
            logger.error(f"Error reading data: {e}")
            return []
    
    def get_scene_info(self, scene_identifier: str) -> Dict[str, Any]:
        """
        Get basic scene information.
        
        Args:
            scene_identifier: Scene token or serial number
            
        Returns:
            Scene information dictionary
        """
        scene_data = self.load_scene_data(scene_identifier)
        
        if not scene_data:
            return {}
        
        return {
            'scene_token': scene_data.get('scene_token'),
            'scene_name': scene_data.get('scene_name'),
            'scene_description': scene_data.get('scene_description'),
            'nbr_samples': scene_data.get('nbr_samples'),
            'nbr_keyframes': len(scene_data.get('key_frames', {})),
            'first_sample_token': scene_data.get('first_sample_token'),
            'last_sample_token': scene_data.get('last_sample_token')
        } 