"""
Unified Data Service

Provides a centralized interface for data access across the entire application.
Consolidates functionality from DataLoader and ContextRetriever to reduce duplication.
"""

from typing import Dict, List, Any, Union, Optional
import numpy as np
from pathlib import Path
from loguru import logger

from .data_loader import DataLoader


class DataService:
    """Unified data service for all data access needs"""
    
    def __init__(self, data_path: str = "data/concatenated_data/concatenated_data.json"):
        """
        Initialize the data service.
        
        Args:
            data_path: Path to the concatenated JSON data file
        """
        self.data_loader = DataLoader(data_path)
        self._analysis_cachegit : Dict[str, Any] = {}
    
    def get_scene_data(self, scene_id: Union[int, str]) -> Dict[str, Any]:
        """
        Get scene data with caching.
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            Scene data dictionary
        """
        return self.data_loader.load_scene_data(scene_id)
    
    def get_keyframe_data(self, scene_id: Union[int, str], keyframe_id: Union[int, str]) -> Dict[str, Any]:
        """
        Get comprehensive data for a specific keyframe.
        
        Args:
            scene_id: Scene identifier
            keyframe_id: Keyframe identifier
            
        Returns:
            Dictionary containing combined keyframe data
        """
        return self.data_loader.get_keyframe_data(scene_id, keyframe_id)
    
    def get_movement_data(self, scene_id: Union[int, str]) -> Dict[str, Any]:
        """
        Get ego vehicle movement data for analysis.
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            Dictionary containing ego movement data
        """
        return self.data_loader.extract_ego_movement_data(scene_id)
    
    def get_qa_pairs(self, scene_id: Union[int, str], keyframe_id: Union[int, str] = 0) -> List[Dict[str, Any]]:
        """
        Get QA pairs for a scene and optionally specific keyframe.
        
        Args:
            scene_id: Scene identifier
            keyframe_id: Keyframe identifier (0 for all keyframes)
            
        Returns:
            List of QA pairs
        """
        return self.data_loader.extract_questions_from_keyframe(scene_id, keyframe_id)
    
    def get_qa_pair(self, scene_id: Union[int, str], keyframe_id: Union[int, str], qa_type: str, qa_serial: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific QA pair.
        
        Args:
            scene_id: Scene identifier
            keyframe_id: Keyframe identifier
            qa_type: Type of QA
            qa_serial: QA pair serial number
            
        Returns:
            QA pair dictionary or None if not found
        """
        try:
            scene_data = self.get_scene_data(scene_id)
            keyframe_token = self.data_loader._assign_keyframe_token(scene_id, keyframe_id)
            
            if keyframe_token not in scene_data['key_frames']:
                return None
            
            keyframe_data = scene_data['key_frames'][keyframe_token]
            qa_data = keyframe_data.get('QA', {})
            
            if qa_type not in qa_data:
                return None
            
            qa_pairs = qa_data[qa_type]
            if qa_serial < 1 or qa_serial > len(qa_pairs):
                return None
            
            return qa_pairs[qa_serial - 1]
            
        except Exception as e:
            logger.error(f"Error getting QA pair: {e}")
            return None
    
    def get_context_for_keyframe(self, scene_id: Union[int, str], keyframe_id: Union[int, str]) -> Dict[str, Any]:
        """
        Get context data for a specific keyframe.
        
        Args:
            scene_id: Scene identifier
            keyframe_id: Keyframe identifier
            
        Returns:
            Context data dictionary
        """
        try:
            scene_data = self.get_scene_data(scene_id)
            keyframe_token = self.data_loader._assign_keyframe_token(scene_id, keyframe_id)
            
            # Create context with just the specific sample and keyframe
            context_data = scene_data.copy()
            context_data['samples'] = {keyframe_token: scene_data['samples'][keyframe_token]}
            context_data['key_frames'] = {keyframe_token: scene_data['key_frames'][keyframe_token]}
            
            return context_data
            
        except Exception as e:
            logger.error(f"Error getting context for keyframe: {e}")
            return {}
    
    def get_vehicle_data_upto_keyframe(self, scene_id: Union[int, str], keyframe_id: Union[int, str]) -> Dict[str, Any]:
        """
        Get vehicle movement data up to a specific keyframe.
        
        Args:
            scene_id: Scene identifier
            keyframe_id: Keyframe identifier
            
        Returns:
            Vehicle movement data dictionary
        """
        try:
            scene_data = self.get_scene_data(scene_id)
            keyframe_token = self.data_loader._assign_keyframe_token(scene_id, keyframe_id)
            
            # Get all samples up to the keyframe
            samples = scene_data['samples']
            sorted_samples = sorted(samples.items(), key=lambda x: x[1]['ego_pose']['timestamp'])
            
            movement_data = []
            timestamps = []
            positions = []
            rotations = []
            
            for sample_token, sample_data in sorted_samples:
                # Stop when we reach the target sample token
                if sample_token == keyframe_token:
                    break
                
                ego_pose = sample_data['ego_pose']
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
                    'velocity': [0, 0, 0],
                    'speed': 0.0,
                    'acceleration': [0, 0, 0],
                    'angular_velocity': 0.0,
                    'curvature': 0.0,
                    'sample_token': sample_token
                }
                
                movement_data.append(movement_entry)
            
            # Calculate derived metrics
            self._calculate_velocity_and_acceleration(movement_data, timestamps)
            self._calculate_curvature(movement_data)
            
            # Calculate summary statistics
            summary_stats = self._calculate_movement_summary(movement_data)
            
            return {
                'scene_id': scene_id,
                'target_sample_token': keyframe_token,
                'nbr_samples': len(movement_data),
                'total_distance': f"{float(summary_stats['total_distance']):.2f} m",
                'total_duration': f"{float(summary_stats['total_duration']):.2f} s",
                'avg_speed': f"{float(summary_stats['avg_speed']):.2f} m/s",
                'max_speed': f"{float(summary_stats['max_speed']):.2f} m/s",
                'avg_acceleration': f"{float(summary_stats['avg_acceleration']):.2f} m/s²",
                'max_acceleration': f"{float(summary_stats['max_acceleration']):.2f} m/s²",
                'avg_curvature': f"{float(summary_stats['avg_curvature']):.2f} rad/m",
                'turning_segments': len(summary_stats['turning_segments']),
                'straight_segments': len(summary_stats['straight_segments']),
                'stopping_periods': len(summary_stats['stopping_periods'])
            }
            
        except Exception as e:
            logger.error(f"Error getting vehicle data up to keyframe: {e}")
            return {}
    
    def get_sensor_data_for_keyframe(self, scene_id: Union[int, str], keyframe_id: Union[int, str]) -> Dict[str, Any]:
        """
        Get sensor data for a specific keyframe.
        
        Args:
            scene_id: Scene identifier
            keyframe_id: Keyframe identifier
            
        Returns:
            Sensor data dictionary
        """
        try:
            scene_data = self.get_scene_data(scene_id)
            keyframe_token = self.data_loader._assign_keyframe_token(scene_id, keyframe_id)
            
            if keyframe_token not in scene_data['samples']:
                return {}
            
            sample_data = scene_data['samples'][keyframe_token]
            annotations = sample_data['annotations']
            sensor_data = sample_data['sensor_data']
            
            # Process annotations to get object detection summary
            category_counts = {}
            sensor_detection_stats = {'lidar_detections': 0, 'radar_detections': 0}
            visibility_stats = {}
            
            for annotation in annotations:
                category = annotation['category']
                category_counts[category] = category_counts.get(category, 0) + 1
                
                # Count sensor detections
                sensor_detection_stats['lidar_detections'] += annotation['num_lidar_pts']
                sensor_detection_stats['radar_detections'] += annotation['num_radar_pts']
                
                # Track visibility levels
                visibility_level = annotation['visibility']['level']
                visibility_stats[visibility_level] = visibility_stats.get(visibility_level, 0) + 1
            
            return {
                'scene_id': scene_id,
                'sample_token': keyframe_token,
                'timestamp': sample_data['timestamp'],
                'object_detection': {
                    'total_objects': len(annotations),
                    'category_counts': category_counts,
                    'unique_categories': list(category_counts.keys()),
                    'num_categories': len(category_counts)
                },
                'sensor_detections': {
                    'total_lidar_points': sensor_detection_stats['lidar_detections'],
                    'total_radar_points': sensor_detection_stats['radar_detections'],
                    'objects_with_lidar': len([a for a in annotations if a['num_lidar_pts'] > 0]),
                    'objects_with_radar': len([a for a in annotations if a['num_radar_pts'] > 0])
                },
                'visibility_distribution': visibility_stats,
                'available_sensors': {
                    'sensor_types': list(sensor_data.keys()),
                    'camera_sensors': [s for s in sensor_data.keys() if 'CAM_' in s],
                    'radar_sensors': [s for s in sensor_data.keys() if 'RADAR_' in s],
                    'lidar_sensors': [s for s in sensor_data.keys() if 'LIDAR_' in s]
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting sensor data for keyframe: {e}")
            return {}
    
    def get_available_scenes(self) -> List[int]:
        """
        Get list of available scene IDs.
        
        Returns:
            List of available scene IDs
        """
        return self.data_loader.get_available_scenes()
    
    def clear_cache(self) -> None:
        """Clear the analysis cache"""
        self._analysis_cache.clear()
        logger.info("Data service cache cleared")
    
    def get_cached_result(self, key: str) -> Optional[Any]:
        """
        Get cached result.
        
        Args:
            key: Cache key
            
        Returns:
            Cached result or None if not found
        """
        return self._analysis_cache.get(key)
    
    def set_cached_result(self, key: str, result: Any) -> None:
        """
        Cache result.
        
        Args:
            key: Cache key
            result: Result to cache
        """
        self._analysis_cache[key] = result
    
    # Helper methods for movement calculations
    def _quaternion_to_heading(self, quaternion: List[float]) -> float:
        """Convert quaternion to heading angle in radians"""
        w, x, y, z = quaternion
        heading = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return heading
    
    def _calculate_velocity_and_acceleration(self, movement_data: List[Dict], timestamps: List[int]):
        """Calculate velocity and acceleration from position data"""
        for i in range(len(movement_data)):
            if i == 0:
                continue
            
            dt = (timestamps[i] - timestamps[i-1]) / 1e6
            
            if dt > 0:
                curr_pos = np.array(movement_data[i]['position'])
                prev_pos = np.array(movement_data[i-1]['position'])
                velocity = (curr_pos - prev_pos) / dt
                movement_data[i]['velocity'] = velocity.tolist()
                movement_data[i]['speed'] = np.linalg.norm(velocity)
                
                curr_heading = movement_data[i]['heading']
                prev_heading = movement_data[i-1]['heading']
                angular_velocity = (curr_heading - prev_heading) / dt
                movement_data[i]['angular_velocity'] = angular_velocity
                
                if i > 1:
                    curr_vel = np.array(movement_data[i]['velocity'])
                    prev_vel = np.array(movement_data[i-1]['velocity'])
                    acceleration = (curr_vel - prev_vel) / dt
                    movement_data[i]['acceleration'] = acceleration.tolist()
    
    def _calculate_curvature(self, movement_data: List[Dict]):
        """Calculate path curvature from heading changes"""
        for i in range(1, len(movement_data)):
            if i == 0:
                continue
            
            heading_change = abs(movement_data[i]['heading'] - movement_data[i-1]['heading'])
            curr_pos = np.array(movement_data[i]['position'])
            prev_pos = np.array(movement_data[i-1]['position'])
            distance = np.linalg.norm(curr_pos - prev_pos)
            
            if distance > 0:
                curvature = heading_change / distance
                movement_data[i]['curvature'] = curvature
            else:
                movement_data[i]['curvature'] = 0.0
    
    def _calculate_movement_summary(self, movement_data: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics for movement data"""
        if not movement_data:
            return {}
        
        speeds = [entry['speed'] for entry in movement_data if entry['speed'] > 0]
        accelerations = [np.linalg.norm(entry['acceleration']) for entry in movement_data if any(entry['acceleration'])]
        curvatures = [entry['curvature'] for entry in movement_data if entry['curvature'] > 0]
        
        total_distance = 0.0
        for i in range(1, len(movement_data)):
            curr_pos = np.array(movement_data[i]['position'])
            prev_pos = np.array(movement_data[i-1]['position'])
            total_distance += np.linalg.norm(curr_pos - prev_pos)
        
        if len(movement_data) > 1:
            total_duration = (movement_data[-1]['timestamp'] - movement_data[0]['timestamp']) / 1e6
        else:
            total_duration = 0.0
        
        turning_segments = []
        straight_segments = []
        stopping_periods = []
        
        for i, entry in enumerate(movement_data):
            if entry['curvature'] > 0.01:
                turning_segments.append(i)
            elif entry['speed'] < 0.5:
                stopping_periods.append(i)
            else:
                straight_segments.append(i)
        
        return {
            'total_distance': total_distance,
            'total_duration': total_duration,
            'avg_speed': np.mean(speeds) if speeds else 0.0,
            'max_speed': np.max(speeds) if speeds else 0.0,
            'avg_acceleration': np.mean(accelerations) if accelerations else 0.0,
            'max_acceleration': np.max(accelerations) if accelerations else 0.0,
            'avg_curvature': np.mean(curvatures) if curvatures else 0.0,
            'turning_segments': turning_segments,
            'straight_segments': straight_segments,
            'stopping_periods': stopping_periods
        } 