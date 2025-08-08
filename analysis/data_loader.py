"""
Data Loader for concatenated Dataset

Loads and parses concatenated JSON data for analysis.
"""

import json
import numpy as np
from typing import Dict, List, Any, Union
from pathlib import Path
from loguru import logger


class DataLoader:
    """Load and parse concatenated JSON data"""
    
    def __init__(self, data_path: str = "data/concatenated_data/concatenated_data.json"):
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
    def _assign_keyframe_token(self, scene_id: Union[int, str], keyframe_id: Union[int, str]) -> str:
        """
        Assign keyframe token to the keyframe id.
        
        Args:
            scene_id: Scene identifier (int or str)
            keyframe_id: Keyframe identifier (int or str)
            
        Returns:
            Keyframe token string
            
        Raises:
            ValueError: If keyframe_id is invalid or not found
        """
        try:
            scene_data = self.load_scene_data(scene_id)
            keyframe_tokens = list(scene_data['key_frames'].keys())
            keyframe_serial_numbers = list(range(1, len(keyframe_tokens) + 1))  
            serial_number_to_keyframe_token_map = dict(zip(keyframe_serial_numbers, keyframe_tokens))
            
            if isinstance(keyframe_id, int):
                if keyframe_id in serial_number_to_keyframe_token_map:
                    return serial_number_to_keyframe_token_map[keyframe_id]
                else:
                    raise ValueError(f"Keyframe ID {keyframe_id} not found. Valid range: 1 to {len(keyframe_tokens)}")
            elif isinstance(keyframe_id, str):
                if keyframe_id in keyframe_tokens:
                    return keyframe_id
                else:
                    raise ValueError(f"Keyframe token '{keyframe_id}' not found in scene data")
            else:
                raise ValueError(f"Invalid keyframe_id type: {type(keyframe_id)}. Expected int or str")
                
        except ValueError:
            # Re-raise ValueError as-is
            raise
        except Exception as e:
            logger.error(f"Error assigning keyframe token: {e}")
            raise ValueError(f"Failed to assign keyframe token for scene {scene_id}, keyframe {keyframe_id}: {e}")
    def extract_questions_from_keyframe(self, scene_id: int, keyframe_id: int) -> List[Dict[str, Any]]:
        """Extract all questions from given keyframe"""
        scene_data = self.load_scene_data(scene_id)
        # base case: if keyframe_id is 0, return all questions
        if keyframe_id != 0:
            # logger.info(f"Extracting questions from keyframe {keyframe_id}")
            qa_pairs = self.load_scene_data(scene_id)["key_frames"][self._assign_keyframe_token(scene_id, keyframe_id)]["QA"]
            return qa_pairs
        
        all_qa_pairs = {}
        for i in range(1, len(scene_data["key_frames"]) + 1):
            # logger.info(f"Extracting questions from keyframe {i}")
            qa_pairs = self.extract_questions_from_keyframe(scene_id, i)
            all_qa_pairs[self._assign_keyframe_token(scene_id, i)] = qa_pairs
        return all_qa_pairs
    def get_keyframe_info_for_scene(self, scene_id: Union[int, str]) -> Dict[str, Any]:
        """
        Get keyframe info for a specific scene.
        """
        info = {"keyframe_tokens": [], "total_keyframes": 0}
        scene_data = self.load_scene_data(scene_id)
        for keyframe_id in scene_data["key_frames"]:
            info["keyframe_tokens"].append(keyframe_id)
            info["total_keyframes"] += 1
        return info
    def get_keyframe_data(self, scene_id: Union[int, str], keyframe_id: Union[int, str]) -> Dict[str, Any]:
        """
        Get comprehensive data for a specific keyframe including both nuScenes sample data and DriveLM keyframe data.
        
        Args:
            scene_id: Scene identifier (int or str)
            keyframe_id: Keyframe identifier (int or str)
            
        Returns:
            Dictionary containing combined keyframe data with the following structure:
            {
                'scene_info': {
                    'scene_name': str,
                    'scene_description': str,
                    'scene_token': str,
                    'keyframe_token': str
                },
                'nuScenes_data': {
                    'timestamp': int,
                    'sensor_data': {
                        'CAM_FRONT': {...},
                        'CAM_BACK': {...},
                        'LIDAR_TOP': {...},
                        'RADAR_FRONT': {...},
                        ...
                    },
                    'annotations': [...],
                    'ego_pose': {...},
                    'prev': str,
                    'next': str
                },
                'DriveLM_data': {
                    'QA': {
                        'perception': [...],
                        'prediction': [...],
                        'planning': [...],
                        'behavior': [...]
                    },
                    'image_paths': {
                        'CAM_FRONT': str,
                        'CAM_BACK': str,
                        ...
                    },
                    'key_object_infos': {
                        '<camera,view,x,y>': {
                            'Category': str,
                            'Status': str,
                            'Visual_description': str,
                            '2d_bbox': [x1, y1, x2, y2]
                        },
                        ...
                    }
                }
            }
        """
        try:
            # Get scene data
            scene_data = self.load_scene_data(scene_id)
            scene_token = self._assign_scene_token(scene_id)
            keyframe_token = self._assign_keyframe_token(scene_token, keyframe_id)
            
            # Extract scene info
            scene_info = {
                'scene_name': scene_data['scene_name'],
                'scene_description': scene_data['scene_description'],
                'scene_token': scene_token,
                'keyframe_token': keyframe_token
            }
            
            # Extract nuScenes sample data
            nuScenes_data = scene_data['samples'][keyframe_token]
            
            # Extract DriveLM keyframe data
            DriveLM_data = scene_data['key_frames'][keyframe_token]
            
            # Combine the data
            combined_data = {
                'scene_info': scene_info,
                'nuScenes_data': nuScenes_data,
                'DriveLM_data': DriveLM_data
            }
            
            # logger.info(f"Successfully loaded keyframe data for scene {scene_id}, keyframe {keyframe_id}")
            return combined_data
            
        except Exception as e:
            logger.error(f"Error loading keyframe data for scene {scene_id}, keyframe {keyframe_id}: {e}")
            raise ValueError(f"Failed to load keyframe data: {e}")   
    def extract_ego_movement_data(self, scene_id: Union[int, str]) -> Dict[str, Any]:
        """
        Extract comprehensive ego vehicle movement data for analysis.
        
        Args:
            scene_id: Scene identifier (int or str)
            
        Returns:
            Dictionary containing ego movement data with the following structure:
            {
                'scene_info': {
                    'scene_name': str,
                    'scene_description': str,
                    'nbr_samples': int
                },
                'movement_data': [
                    {
                        'timestamp': int,
                        'position': [x, y, z],
                        'rotation': [w, x, y, z],
                        'heading': float,
                        'velocity': [vx, vy, vz],
                        'speed': float,
                        'acceleration': [ax, ay, az],
                        'angular_velocity': float,
                        'curvature': float
                    },
                    ...
                ],
                'summary_stats': {
                    'total_distance': float,
                    'avg_speed': float,
                    'max_speed': float,
                    'turning_segments': list,
                    'straight_segments': list,
                    'stopping_periods': list
                }
            }
        """
        try:
            scene_data = self.load_scene_data(scene_id)
            samples = scene_data['samples']
            
            # Extract basic scene info
            scene_info = {
                'scene_name': scene_data['scene_name'],
                'scene_description': scene_data['scene_description'],
                'nbr_samples': scene_data['nbr_samples']
            }
            
            # Extract movement data from samples
            movement_data = []
            timestamps = []
            positions = []
            rotations = []
            
            # Sort samples by timestamp to ensure chronological order
            sorted_samples = sorted(samples.items(), key=lambda x: x[1]['timestamp'])
            
            for sample_token, sample_data in sorted_samples:
                ego_pose = sample_data['ego_pose']
                
                # Extract basic pose data
                timestamp = ego_pose['timestamp']
                position = ego_pose['translation']
                rotation = ego_pose['rotation']
                
                # Calculate heading from quaternion
                heading = self._quaternion_to_heading(rotation)
                
                # Store for velocity calculations
                timestamps.append(timestamp)
                positions.append(position)
                rotations.append(rotation)
                
                # Initialize movement data entry
                movement_entry = {
                    'timestamp': timestamp,
                    'position': position,
                    'rotation': rotation,
                    'heading': heading,
                    'velocity': [0, 0, 0],  # Will be calculated
                    'speed': 0.0,
                    'acceleration': [0, 0, 0],  # Will be calculated
                    'angular_velocity': 0.0,
                    'curvature': 0.0
                }
                
                movement_data.append(movement_entry)
            
            # Calculate derived metrics
            self._calculate_velocity_and_acceleration(movement_data, timestamps)
            self._calculate_curvature(movement_data)
            
            # Calculate summary statistics
            summary_stats = self._calculate_movement_summary(movement_data)
            
            return {
                'scene_info': scene_info,
                'movement_data': movement_data,
                'summary_stats': summary_stats
            }
            
        except Exception as e:
            logger.error(f"Error extracting ego movement data: {e}")
            return {}
    def _quaternion_to_heading(self, quaternion: List[float]) -> float:
        """
        Convert quaternion to heading angle in radians.
        
        Args:
            quaternion: [w, x, y, z] quaternion
            
        Returns:
            Heading angle in radians
        """
        w, x, y, z = quaternion
        # Extract yaw from quaternion
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return yaw
    def _calculate_velocity_and_acceleration(self, movement_data: List[Dict], timestamps: List[int]):
        """
        Calculate velocity and acceleration for movement data.
        
        Args:
            movement_data: List of movement data dictionaries
            timestamps: List of timestamps
        """
        for i in range(1, len(movement_data)):
            # Time difference in seconds (timestamps are in microseconds)
            dt = (timestamps[i] - timestamps[i-1]) / 1e6
            
            if dt > 0:
                # Calculate velocity
                pos_curr = np.array(movement_data[i]['position'])
                pos_prev = np.array(movement_data[i-1]['position'])
                velocity = (pos_curr - pos_prev) / dt
                
                movement_data[i]['velocity'] = velocity.tolist()
                movement_data[i]['speed'] = np.linalg.norm(velocity)
                
                # Calculate acceleration (if we have previous velocity)
                if i > 1:
                    vel_prev = np.array(movement_data[i-1]['velocity'])
                    acceleration = (velocity - vel_prev) / dt
                    movement_data[i]['acceleration'] = acceleration.tolist()
                
                # Calculate angular velocity
                heading_curr = movement_data[i]['heading']
                heading_prev = movement_data[i-1]['heading']
                angular_velocity = (heading_curr - heading_prev) / dt
                movement_data[i]['angular_velocity'] = angular_velocity
    def _calculate_curvature(self, movement_data: List[Dict]):
        """
        Calculate trajectory curvature for movement data.
        
        Args:
            movement_data: List of movement data dictionaries
        """
        for i in range(1, len(movement_data) - 1):
            # Get three consecutive points
            pos_prev = np.array(movement_data[i-1]['position'][:2])  # Use only x,y
            pos_curr = np.array(movement_data[i]['position'][:2])
            pos_next = np.array(movement_data[i+1]['position'][:2])
            
            # Calculate curvature using three-point method
            if np.linalg.norm(pos_next - pos_prev) > 0:
                # Vector from prev to curr
                v1 = pos_curr - pos_prev
                # Vector from curr to next
                v2 = pos_next - pos_curr
                
                # Cross product magnitude
                cross_mag = abs(v1[0] * v2[1] - v1[1] * v2[0])
                
                # Curvature = cross_mag / (|v1| * |v2| * |v1 + v2|)
                v1_mag = np.linalg.norm(v1)
                v2_mag = np.linalg.norm(v2)
                v_sum_mag = np.linalg.norm(v1 + v2)
                
                if v1_mag * v2_mag * v_sum_mag > 0:
                    curvature = cross_mag / (v1_mag * v2_mag * v_sum_mag)
                    movement_data[i]['curvature'] = curvature
    def _calculate_movement_summary(self, movement_data: List[Dict]) -> Dict[str, Any]:
        """
        Calculate summary statistics for movement data.
        
        Args:
            movement_data: List of movement data dictionaries
            
        Returns:
            Dictionary with summary statistics
        """
        if not movement_data:
            return {}
        
        speeds = [entry['speed'] for entry in movement_data if entry['speed'] > 0]
        curvatures = [entry['curvature'] for entry in movement_data if entry['curvature'] > 0]
        
        # Calculate total distance
        total_distance = 0
        for i in range(1, len(movement_data)):
            pos_curr = np.array(movement_data[i]['position'])
            pos_prev = np.array(movement_data[i-1]['position'])
            total_distance += np.linalg.norm(pos_curr - pos_prev)
        
        # Identify movement segments
        turning_segments = []
        straight_segments = []
        stopping_periods = []
        
        # Simple threshold-based segmentation
        curvature_threshold = 0.01
        speed_threshold = 0.5  # m/s
        
        current_segment_start = 0
        current_segment_type = None
        
        for i, entry in enumerate(movement_data):
            if entry['curvature'] > curvature_threshold:
                segment_type = 'turning'
            elif entry['speed'] < speed_threshold:
                segment_type = 'stopping'
            else:
                segment_type = 'straight'
            
            if segment_type != current_segment_type:
                # End current segment
                if current_segment_type == 'turning':
                    turning_segments.append((current_segment_start, i-1))
                elif current_segment_type == 'straight':
                    straight_segments.append((current_segment_start, i-1))
                elif current_segment_type == 'stopping':
                    stopping_periods.append((current_segment_start, i-1))
                
                # Start new segment
                current_segment_start = i
                current_segment_type = segment_type
        
        return {
            'total_distance': total_distance,
            'avg_speed': np.mean(speeds) if speeds else 0.0,
            'max_speed': np.max(speeds) if speeds else 0.0,
            'min_speed': np.min(speeds) if speeds else 0.0,
            'avg_curvature': np.mean(curvatures) if curvatures else 0.0,
            'max_curvature': np.max(curvatures) if curvatures else 0.0,
            'turning_segments': turning_segments,
            'straight_segments': straight_segments,
            'stopping_periods': stopping_periods,
            'total_duration': (movement_data[-1]['timestamp'] - movement_data[0]['timestamp']) / 1e6  # seconds
        }
    