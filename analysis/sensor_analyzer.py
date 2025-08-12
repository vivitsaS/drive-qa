"""
Sensor Data Analyzer

Analyzes camera and sensor data patterns across scenes:
- Sensor coverage analysis
- Scene-specific sensor usage
- Multi-modal sensor fusion patterns
"""

from typing import Dict, Any
from collections import defaultdict
from loguru import logger

from parsers.data_loader import DataLoader


class SensorAnalyzer:
    """Analyze sensor data patterns and coverage"""
    
    def __init__(self, data_loader: DataLoader):
        """Initialize the sensor analyzer"""
        self.data_loader = data_loader
        self.cameras = ['CAM_FRONT', 'CAM_BACK', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        self.radars = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
        self.lidars = ['LIDAR_TOP']
        self.all_sensors = self.cameras + self.radars + self.lidars
    
    def analyze_sensor_coverage(self) -> Dict[str, Any]:
        """
        Analyze sensor coverage patterns across all scenes.
        
        Returns:
            Dictionary containing sensor coverage analysis
        """
        logger.info("Analyzing sensor coverage patterns...")
        
        coverage_data = {
            'camera_activity': {},
            'sensor_availability': {},
            'missing_data': {},
            'sensor_fusion_patterns': {}
        }
        
        # Analyze each scene
        for scene_id in range(1, 7):
            scene_name = f"Scene {scene_id}"
            scene_data = self.data_loader.load_scene_data(scene_id)
            
            # Camera activity analysis
            camera_activity = self._analyze_camera_activity(scene_data)
            coverage_data['camera_activity'][scene_name] = camera_activity
            
            # Sensor availability analysis
            sensor_availability = self._analyze_sensor_availability(scene_data)
            coverage_data['sensor_availability'][scene_name] = sensor_availability
            
            # Missing data detection
            missing_data = self._detect_missing_data(scene_data)
            coverage_data['missing_data'][scene_name] = missing_data
            
            # Multi-modal sensor fusion patterns
            fusion_patterns = self._analyze_sensor_fusion(scene_data)
            coverage_data['sensor_fusion_patterns'][scene_name] = fusion_patterns
        
        return coverage_data
    
    def analyze_scene_specific_usage(self) -> Dict[str, Any]:
        """
        Analyze scene-specific sensor usage patterns.
        
        Returns:
            Dictionary containing scene-specific sensor analysis
        """
        logger.info("Analyzing scene-specific sensor usage...")
        
        scene_usage_data = {
            'camera_importance': {},
            'sensor_redundancy': {},
            'critical_sensors': {}
        }
        
        # Analyze each scene
        for scene_id in range(1, 7):
            scene_name = f"Scene {scene_id}"
            scene_data = self.data_loader.load_scene_data(scene_id)
            
            # Camera importance by scene type
            camera_importance = self._analyze_camera_importance(scene_data)
            scene_usage_data['camera_importance'][scene_name] = camera_importance
            
            # Sensor redundancy analysis
            sensor_redundancy = self._analyze_sensor_redundancy(scene_data)
            scene_usage_data['sensor_redundancy'][scene_name] = sensor_redundancy
            
            # Critical sensor identification
            critical_sensors = self._identify_critical_sensors(scene_data)
            scene_usage_data['critical_sensors'][scene_name] = critical_sensors
        
        return scene_usage_data
    
    def _analyze_camera_activity(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze camera activity patterns in a scene"""
        camera_counts = {camera: 0 for camera in self.cameras}
        total_samples = 0
        
        for sample_token, sample_data in scene_data['samples'].items():
            total_samples += 1
            sensor_data = sample_data.get('sensor_data', {})
            
            for camera in self.cameras:
                if camera in sensor_data:
                    camera_counts[camera] += 1
        
        # Calculate activity percentages
        camera_activity = {}
        for camera, count in camera_counts.items():
            activity_percentage = (count / total_samples * 100) if total_samples > 0 else 0
            camera_activity[camera] = {
                'count': count,
                'percentage': activity_percentage,
                'is_active': activity_percentage > 50  # Consider active if >50% samples have data
            }
        
        return camera_activity
    
    def _analyze_sensor_availability(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sensor data availability across all sensors"""
        sensor_counts = {sensor: 0 for sensor in self.all_sensors}
        total_samples = 0
        
        for sample_token, sample_data in scene_data['samples'].items():
            total_samples += 1
            sensor_data = sample_data.get('sensor_data', {})
            
            for sensor in self.all_sensors:
                if sensor in sensor_data:
                    sensor_counts[sensor] += 1
        
        # Calculate availability percentages
        sensor_availability = {}
        for sensor, count in sensor_counts.items():
            availability_percentage = (count / total_samples * 100) if total_samples > 0 else 0
            sensor_availability[sensor] = {
                'count': count,
                'percentage': availability_percentage,
                'is_available': availability_percentage > 80  # Consider available if >80% samples have data
            }
        
        return sensor_availability
    
    def _detect_missing_data(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect missing sensor data patterns"""
        missing_patterns = {
            'samples_with_missing_data': 0,
            'most_missing_sensor': None,
            'missing_data_percentage': 0
        }
        
        total_samples = 0
        missing_sensor_counts = {sensor: 0 for sensor in self.all_sensors}
        
        for sample_token, sample_data in scene_data['samples'].items():
            total_samples += 1
            sensor_data = sample_data.get('sensor_data', {})
            
            # Check for missing sensors
            missing_sensors = [sensor for sensor in self.all_sensors if sensor not in sensor_data]
            if missing_sensors:
                missing_patterns['samples_with_missing_data'] += 1
                for sensor in missing_sensors:
                    missing_sensor_counts[sensor] += 1
        
        # Find most missing sensor
        if missing_sensor_counts:
            most_missing = max(missing_sensor_counts.items(), key=lambda x: x[1])
            missing_patterns['most_missing_sensor'] = most_missing[0]
            missing_patterns['missing_data_percentage'] = (missing_patterns['samples_with_missing_data'] / total_samples * 100) if total_samples > 0 else 0
        
        return missing_patterns
    
    def _analyze_sensor_fusion(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze multi-modal sensor fusion patterns"""
        fusion_patterns = {
            'camera_radar_fusion': 0,
            'camera_lidar_fusion': 0,
            'full_sensor_fusion': 0,
            'total_samples': 0
        }
        
        for sample_token, sample_data in scene_data['samples'].items():
            fusion_patterns['total_samples'] += 1
            sensor_data = sample_data.get('sensor_data', {})
            
            # Check for camera-radar fusion
            has_camera = any(camera in sensor_data for camera in self.cameras)
            has_radar = any(radar in sensor_data for radar in self.radars)
            if has_camera and has_radar:
                fusion_patterns['camera_radar_fusion'] += 1
            
            # Check for camera-lidar fusion
            has_lidar = any(lidar in sensor_data for lidar in self.lidars)
            if has_camera and has_lidar:
                fusion_patterns['camera_lidar_fusion'] += 1
            
            # Check for full sensor fusion
            if has_camera and has_radar and has_lidar:
                fusion_patterns['full_sensor_fusion'] += 1
        
        # Calculate percentages
        total = fusion_patterns['total_samples']
        if total > 0:
            fusion_patterns['camera_radar_fusion_pct'] = fusion_patterns['camera_radar_fusion'] / total * 100
            fusion_patterns['camera_lidar_fusion_pct'] = fusion_patterns['camera_lidar_fusion'] / total * 100
            fusion_patterns['full_sensor_fusion_pct'] = fusion_patterns['full_sensor_fusion'] / total * 100
        
        return fusion_patterns
    
    def _analyze_camera_importance(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze camera importance by scene type"""
        camera_importance = {}
        
        # Get scene description to understand scene type
        scene_description = scene_data.get('scene_description', '').lower()
        
        # Define importance based on scene characteristics
        for camera in self.cameras:
            importance_score = 0
            
            # Front camera importance
            if 'CAM_FRONT' in camera:
                importance_score += 3  # Always important
            
            # Back camera importance
            if 'CAM_BACK' in camera:
                if any(word in scene_description for word in ['parking', 'reverse', 'backing']):
                    importance_score += 3
                else:
                    importance_score += 1
            
            # Side cameras importance
            if any(side in camera for side in ['LEFT', 'RIGHT']):
                if any(word in scene_description for word in ['turn', 'lane', 'merge', 'intersection']):
                    importance_score += 2
                else:
                    importance_score += 1
            
            camera_importance[camera] = {
                'importance_score': importance_score,
                'importance_level': 'high' if importance_score >= 3 else 'medium' if importance_score >= 2 else 'low'
            }
        
        return camera_importance
    
    def _analyze_sensor_redundancy(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sensor redundancy patterns"""
        redundancy_analysis = {
            'camera_redundancy': {},
            'radar_redundancy': {},
            'overall_redundancy': 0
        }
        
        # Analyze camera redundancy
        camera_coverage = defaultdict(int)
        total_samples = 0
        
        for sample_token, sample_data in scene_data['samples'].items():
            total_samples += 1
            sensor_data = sample_data.get('sensor_data', {})
            
            active_cameras = [cam for cam in self.cameras if cam in sensor_data]
            camera_coverage[len(active_cameras)] += 1
        
        # Calculate redundancy metrics
        for num_cameras, count in camera_coverage.items():
            redundancy_analysis['camera_redundancy'][f'{num_cameras}_cameras'] = {
                'count': count,
                'percentage': count / total_samples * 100 if total_samples > 0 else 0
            }
        
        # Overall redundancy (average number of active sensors per sample)
        total_active_sensors = 0
        for sample_token, sample_data in scene_data['samples'].items():
            sensor_data = sample_data.get('sensor_data', {})
            total_active_sensors += len(sensor_data)
        
        redundancy_analysis['overall_redundancy'] = total_active_sensors / total_samples if total_samples > 0 else 0
        
        return redundancy_analysis
    
    def _identify_critical_sensors(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """Identify critical sensors for each scene"""
        critical_sensors = {
            'essential_cameras': [],
            'essential_radars': [],
            'essential_lidars': [],
            'criticality_reason': {}
        }
        
        # Get scene description
        scene_description = scene_data.get('scene_description', '').lower()
        
        # Essential cameras based on scene type
        if any(word in scene_description for word in ['intersection', 'crossing', 'traffic']):
            critical_sensors['essential_cameras'].extend(['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'])
        elif any(word in scene_description for word in ['parking', 'reverse']):
            critical_sensors['essential_cameras'].extend(['CAM_FRONT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'])
        else:
            critical_sensors['essential_cameras'].extend(['CAM_FRONT'])  # Always essential
        
        # Essential radars (front radar is always critical)
        critical_sensors['essential_radars'].append('RADAR_FRONT')
        
        # Essential lidars (top lidar is always critical)
        critical_sensors['essential_lidars'].extend(self.lidars)
        
        # Add reasoning
        critical_sensors['criticality_reason'] = {
            'CAM_FRONT': 'Primary forward vision',
            'RADAR_FRONT': 'Forward collision detection',
            'LIDAR_TOP': '3D environment mapping'
        }
        
        return critical_sensors 