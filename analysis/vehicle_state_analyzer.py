"""
Vehicle State Analyzer

Analyzes ego vehicle movement and state data for driving behavior insights.
"""

import json
from typing import Dict, List, Any, Union, Tuple
from collections import Counter, defaultdict
import numpy as np
from loguru import logger

from parsers.data_loader import DataLoader


class VehicleStateAnalyzer:
    """Vehicle state analyzer for driving behavior insights"""

    def __init__(self, data_loader: DataLoader):
        """Initialize the vehicle state analyzer"""
        self.data_loader = data_loader

    def get_velocity_summary(self, scene_id: Union[int, str]) -> Dict[str, Any]:
        """
        Get basic vehicle state information summary.
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            Dictionary containing velocity and basic state summary
        """
        try:
            movement_data = self.data_loader.extract_ego_movement_data(scene_id)
            if not movement_data:
                return {}
            
            # Extract velocity-related metrics
            speeds = [entry['speed'] for entry in movement_data['movement_data'] if entry['speed'] > 0]
            accelerations = [np.linalg.norm(entry['acceleration']) for entry in movement_data['movement_data'] if any(entry['acceleration'])]
            
            return {
                'avg_speed': np.mean(speeds) if speeds else 0.0,
                'max_speed': np.max(speeds) if speeds else 0.0,
                'min_speed': np.min(speeds) if speeds else 0.0,
                'speed_std': np.std(speeds) if speeds else 0.0,
                'avg_acceleration': np.mean(accelerations) if accelerations else 0.0,
                'max_acceleration': np.max(accelerations) if accelerations else 0.0,
                'total_distance': movement_data['summary_stats']['total_distance'],
                'total_duration': movement_data['summary_stats']['total_duration'],
                'movement_segments': {
                    'turning': len(movement_data['summary_stats']['turning_segments']),
                    'straight': len(movement_data['summary_stats']['straight_segments']),
                    'stopping': len(movement_data['summary_stats']['stopping_periods'])
                }
            }
        except Exception as e:
            logger.error(f"Error getting velocity summary: {e}")
            return {}

    def classify_driving_style(self, scene_id: Union[int, str]) -> Dict[str, Any]:
        """
        Classify driving style as aggressive vs conservative.
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            Dictionary containing driving style classification
        """
        try:
            movement_data = self.data_loader.extract_ego_movement_data(scene_id)
            if not movement_data:
                return {}
            
            speeds = [entry['speed'] for entry in movement_data['movement_data'] if entry['speed'] > 0]
            accelerations = [np.linalg.norm(entry['acceleration']) for entry in movement_data['movement_data'] if any(entry['acceleration'])]
            curvatures = [entry['curvature'] for entry in movement_data['movement_data'] if entry['curvature'] > 0]
            
            # Calculate style indicators
            avg_speed = np.mean(speeds) if speeds else 0.0
            max_speed = np.max(speeds) if speeds else 0.0
            avg_accel = np.mean(accelerations) if accelerations else 0.0
            max_accel = np.max(accelerations) if accelerations else 0.0
            avg_curvature = np.mean(curvatures) if curvatures else 0.0
            
            # Define thresholds for classification
            speed_threshold = 5.0  # m/s
            accel_threshold = 2.0  # m/s²
            curvature_threshold = 0.015
            
            # Calculate style score (0 = conservative, 1 = aggressive)
            speed_score = min(avg_speed / speed_threshold, 1.0)
            accel_score = min(avg_accel / accel_threshold, 1.0)
            curvature_score = min(avg_curvature / curvature_threshold, 1.0)
            
            overall_score = (speed_score + accel_score + curvature_score) / 3
            
            # Classify style
            if overall_score < 0.3:
                style = "conservative"
            elif overall_score < 0.7:
                style = "moderate"
            else:
                style = "aggressive"
            
            return {
                'style': style,
                'overall_score': overall_score,
                'speed_score': speed_score,
                'acceleration_score': accel_score,
                'curvature_score': curvature_score,
                'metrics': {
                    'avg_speed': avg_speed,
                    'max_speed': max_speed,
                    'avg_acceleration': avg_accel,
                    'max_acceleration': max_accel,
                    'avg_curvature': avg_curvature
                }
            }
        except Exception as e:
            logger.error(f"Error classifying driving style: {e}")
            return {}

    def analyze_smoothness(self, scene_id: Union[int, str]) -> Dict[str, Any]:
        """
        Analyze driving smoothness metrics.
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            Dictionary containing smoothness analysis
        """
        try:
            movement_data = self.data_loader.extract_ego_movement_data(scene_id)
            if not movement_data:
                return {}
            
            # Calculate jerk (rate of change of acceleration)
            jerks = []
            angular_accelerations = []
            
            for i in range(2, len(movement_data['movement_data'])):
                curr_accel = np.array(movement_data['movement_data'][i]['acceleration'])
                prev_accel = np.array(movement_data['movement_data'][i-1]['acceleration'])
                
                # Time difference in seconds
                dt = (movement_data['movement_data'][i]['timestamp'] - movement_data['movement_data'][i-1]['timestamp']) / 1e6
                
                if dt > 0:
                    jerk = np.linalg.norm(curr_accel - prev_accel) / dt
                    jerks.append(jerk)
                
                # Angular acceleration
                curr_angular_vel = movement_data['movement_data'][i]['angular_velocity']
                prev_angular_vel = movement_data['movement_data'][i-1]['angular_velocity']
                angular_accel = abs(curr_angular_vel - prev_angular_vel) / dt if dt > 0 else 0
                angular_accelerations.append(angular_accel)
            
            # Calculate smoothness metrics
            avg_jerk = np.mean(jerks) if jerks else 0.0
            max_jerk = np.max(jerks) if jerks else 0.0
            avg_angular_accel = np.mean(angular_accelerations) if angular_accelerations else 0.0
            max_angular_accel = np.max(angular_accelerations) if angular_accelerations else 0.0
            
            # Smoothness score (lower is smoother)
            jerk_score = min(avg_jerk / 5.0, 1.0)  # Normalize to 0-1
            angular_score = min(avg_angular_accel / 2.0, 1.0)
            smoothness_score = 1.0 - (jerk_score + angular_score) / 2
            
            return {
                'smoothness_score': smoothness_score,
                'avg_jerk': avg_jerk,
                'max_jerk': max_jerk,
                'avg_angular_acceleration': avg_angular_accel,
                'max_angular_acceleration': max_angular_accel,
                'smoothness_level': 'smooth' if smoothness_score > 0.7 else 'moderate' if smoothness_score > 0.4 else 'rough'
            }
        except Exception as e:
            logger.error(f"Error analyzing smoothness: {e}")
            return {}

    def analyze_predictability(self, scene_id: Union[int, str]) -> Dict[str, Any]:
        """
        Analyze driving predictability.
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            Dictionary containing predictability analysis
        """
        try:
            movement_data = self.data_loader.extract_ego_movement_data(scene_id)
            if not movement_data:
                return {}
            
            speeds = [entry['speed'] for entry in movement_data['movement_data'] if entry['speed'] > 0]
            accelerations = [np.linalg.norm(entry['acceleration']) for entry in movement_data['movement_data'] if any(entry['acceleration'])]
            curvatures = [entry['curvature'] for entry in movement_data['movement_data'] if entry['curvature'] > 0]
            
            # Calculate consistency metrics
            speed_std = np.std(speeds) if speeds else 0.0
            accel_std = np.std(accelerations) if accelerations else 0.0
            curvature_std = np.std(curvatures) if curvatures else 0.0
            
            # Normalize standard deviations
            speed_consistency = max(0, 1 - (speed_std / 3.0))  # Lower std = higher consistency
            accel_consistency = max(0, 1 - (accel_std / 2.0))
            curvature_consistency = max(0, 1 - (curvature_std / 0.01))
            
            # Overall predictability score
            predictability_score = (speed_consistency + accel_consistency + curvature_consistency) / 3
            
            return {
                'predictability_score': predictability_score,
                'speed_consistency': speed_consistency,
                'acceleration_consistency': accel_consistency,
                'curvature_consistency': curvature_consistency,
                'predictability_level': 'predictable' if predictability_score > 0.7 else 'moderate' if predictability_score > 0.4 else 'unpredictable'
            }
        except Exception as e:
            logger.error(f"Error analyzing predictability: {e}")
            return {}

    def calculate_risk_score(self, scene_id: Union[int, str]) -> Dict[str, Any]:
        """
        Calculate risk score for driving behavior.
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            Dictionary containing risk assessment
        """
        try:
            movement_data = self.data_loader.extract_ego_movement_data(scene_id)
            if not movement_data:
                return {}
            
            speeds = [entry['speed'] for entry in movement_data['movement_data'] if entry['speed'] > 0]
            accelerations = [np.linalg.norm(entry['acceleration']) for entry in movement_data['movement_data'] if any(entry['acceleration'])]
            jerks = []
            
            # Calculate jerk for risk assessment
            for i in range(2, len(movement_data['movement_data'])):
                curr_accel = np.array(movement_data['movement_data'][i]['acceleration'])
                prev_accel = np.array(movement_data['movement_data'][i-1]['acceleration'])
                dt = (movement_data['movement_data'][i]['timestamp'] - movement_data['movement_data'][i-1]['timestamp']) / 1e6
                if dt > 0:
                    jerk = np.linalg.norm(curr_accel - prev_accel) / dt
                    jerks.append(jerk)
            
            # Risk factors
            max_speed = np.max(speeds) if speeds else 0.0
            max_accel = np.max(accelerations) if accelerations else 0.0
            max_jerk = np.max(jerks) if jerks else 0.0
            
            # Risk thresholds
            speed_risk = min(max_speed / 10.0, 1.0)  # 10 m/s threshold
            accel_risk = min(max_accel / 5.0, 1.0)   # 5 m/s² threshold
            jerk_risk = min(max_jerk / 10.0, 1.0)    # 10 m/s³ threshold
            
            # Overall risk score
            risk_score = (speed_risk + accel_risk + jerk_risk) / 3
            
            return {
                'risk_score': risk_score,
                'speed_risk': speed_risk,
                'acceleration_risk': accel_risk,
                'jerk_risk': jerk_risk,
                'risk_level': 'low' if risk_score < 0.3 else 'medium' if risk_score < 0.7 else 'high',
                'max_speed': max_speed,
                'max_acceleration': max_accel,
                'max_jerk': max_jerk
            }
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return {}

    def analyze_safety_margins(self, scene_id: Union[int, str]) -> Dict[str, Any]:
        """
        Analyze safety margins maintained from other objects.
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            Dictionary containing safety margin analysis
        """
        try:
            scene_data = self.data_loader.load_scene_data(scene_id)
            samples = scene_data['samples']
            
            safety_margins = []
            object_interactions = []
            
            for sample_token, sample_data in samples.items():
                ego_pose = sample_data['ego_pose']
                ego_position = np.array(ego_pose['translation'])
                
                # Calculate distances to all objects
                for annotation in sample_data.get('annotations', []):
                    obj_position = np.array(annotation['translation'])
                    distance = np.linalg.norm(ego_position - obj_position)
                    
                    safety_margins.append({
                        'timestamp': ego_pose['timestamp'],
                        'object_category': annotation['category'],
                        'distance': distance,
                        'object_size': annotation['size']
                    })
                    
                    # Identify close interactions
                    if distance < 5.0:  # 5 meters threshold
                        object_interactions.append({
                            'timestamp': ego_pose['timestamp'],
                            'object_category': annotation['category'],
                            'distance': distance,
                            'risk_level': 'high' if distance < 2.0 else 'medium' if distance < 3.0 else 'low'
                        })
            
            # Calculate safety metrics
            distances = [item['distance'] for item in safety_margins]
            avg_distance = np.mean(distances) if distances else 0.0
            min_distance = np.min(distances) if distances else 0.0
            
            return {
                'avg_safety_margin': avg_distance,
                'min_safety_margin': min_distance,
                'close_interactions': len(object_interactions),
                'high_risk_interactions': len([i for i in object_interactions if i['risk_level'] == 'high']),
                'safety_score': max(0, 1 - (len(object_interactions) / 10))  # Fewer interactions = higher safety
            }
        except Exception as e:
            logger.error(f"Error analyzing safety margins: {e}")
            return {}

    def assess_collision_risk(self, scene_id: Union[int, str]) -> Dict[str, Any]:
        """
        Assess collision risk based on object proximity and movement.
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            Dictionary containing collision risk assessment
        """
        try:
            scene_data = self.data_loader.load_scene_data(scene_id)
            samples = scene_data['samples']
            
            collision_risks = []
            
            for sample_token, sample_data in samples.items():
                ego_pose = sample_data['ego_pose']
                ego_position = np.array(ego_pose['translation'])
                ego_velocity = np.array([0, 0, 0])  # Would need to calculate from movement data
                
                for annotation in sample_data.get('annotations', []):
                    obj_position = np.array(annotation['translation'])
                    distance = np.linalg.norm(ego_position - obj_position)
                    
                    # Simple collision risk based on distance and relative velocity
                    # In a real implementation, you'd calculate relative velocity
                    collision_risk = max(0, 1 - (distance / 10.0))  # Risk decreases with distance
                    
                    collision_risks.append({
                        'timestamp': ego_pose['timestamp'],
                        'object_category': annotation['category'],
                        'distance': distance,
                        'collision_risk': collision_risk
                    })
            
            # Calculate overall collision risk
            avg_risk = np.mean([r['collision_risk'] for r in collision_risks]) if collision_risks else 0.0
            max_risk = np.max([r['collision_risk'] for r in collision_risks]) if collision_risks else 0.0
            
            return {
                'avg_collision_risk': avg_risk,
                'max_collision_risk': max_risk,
                'high_risk_objects': len([r for r in collision_risks if r['collision_risk'] > 0.5]),
                'risk_level': 'low' if avg_risk < 0.2 else 'medium' if avg_risk < 0.5 else 'high'
            }
        except Exception as e:
            logger.error(f"Error assessing collision risk: {e}")
            return {}

    def analyze_traffic_compliance(self, scene_id: Union[int, str]) -> Dict[str, Any]:
        """
        Analyze compliance with traffic rules.
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            Dictionary containing traffic compliance analysis
        """
        try:
            movement_data = self.data_loader.extract_ego_movement_data(scene_id)
            if not movement_data:
                return {}
            
            speeds = [entry['speed'] for entry in movement_data['movement_data'] if entry['speed'] > 0]
            
            # Define traffic rule thresholds (example values)
            speed_limit = 8.0  # m/s (about 29 km/h)
            max_accel_limit = 3.0  # m/s²
            
            # Check speed compliance
            speed_violations = [s for s in speeds if s > speed_limit]
            speed_compliance_rate = 1 - (len(speed_violations) / len(speeds)) if speeds else 1.0
            
            # Check acceleration compliance
            accelerations = [np.linalg.norm(entry['acceleration']) for entry in movement_data['movement_data'] if any(entry['acceleration'])]
            accel_violations = [a for a in accelerations if a > max_accel_limit]
            accel_compliance_rate = 1 - (len(accel_violations) / len(accelerations)) if accelerations else 1.0
            
            # Overall compliance score
            compliance_score = (speed_compliance_rate + accel_compliance_rate) / 2
            
            return {
                'compliance_score': compliance_score,
                'speed_compliance_rate': speed_compliance_rate,
                'acceleration_compliance_rate': accel_compliance_rate,
                'speed_violations': len(speed_violations),
                'acceleration_violations': len(accel_violations),
                'compliance_level': 'good' if compliance_score > 0.8 else 'moderate' if compliance_score > 0.6 else 'poor'
            }
        except Exception as e:
            logger.error(f"Error analyzing traffic compliance: {e}")
            return {}

    def detect_system_performance_issues(self, scene_id: Union[int, str]) -> Dict[str, Any]:
        """
        Detect sensor or tracking performance issues.
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            Dictionary containing system performance analysis
        """
        try:
            scene_data = self.data_loader.load_scene_data(scene_id)
            samples = scene_data['samples']
            
            issues = []
            
            # Check for missing or inconsistent data
            for sample_token, sample_data in samples.items():
                ego_pose = sample_data['ego_pose']
                
                # Check for missing pose data
                if not ego_pose.get('translation') or not ego_pose.get('rotation'):
                    issues.append({
                        'type': 'missing_pose_data',
                        'timestamp': ego_pose.get('timestamp'),
                        'severity': 'high'
                    })
                
                # Check for sensor data issues
                sensor_data = sample_data.get('sensor_data', {})
                expected_sensors = ['CAM_FRONT', 'LIDAR_TOP']
                
                for sensor in expected_sensors:
                    if sensor not in sensor_data:
                        issues.append({
                            'type': 'missing_sensor_data',
                            'sensor': sensor,
                            'timestamp': ego_pose.get('timestamp'),
                            'severity': 'medium'
                        })
                
                # Check for annotation consistency
                annotations = sample_data.get('annotations', [])
                if len(annotations) == 0:
                    issues.append({
                        'type': 'no_annotations',
                        'timestamp': ego_pose.get('timestamp'),
                        'severity': 'medium'
                    })
            
            return {
                'total_issues': len(issues),
                'high_severity_issues': len([i for i in issues if i['severity'] == 'high']),
                'medium_severity_issues': len([i for i in issues if i['severity'] == 'medium']),
                'issues': issues,
                'system_health': 'good' if len(issues) == 0 else 'moderate' if len(issues) < 5 else 'poor'
            }
        except Exception as e:
            logger.error(f"Error detecting system performance issues: {e}")
            return {}


    def analyze_scene(self, scene_id: Union[int, str]) -> Dict[str, Any]:
        """
        Generate comprehensive analysis combining all metrics.
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            Dictionary containing comprehensive analysis
        """
        try:
            analysis = {
                'scene_id': scene_id,
                'velocity_summary': self.get_velocity_summary(scene_id),
                'driving_style': self.classify_driving_style(scene_id),
                'smoothness': self.analyze_smoothness(scene_id),
                'predictability': self.analyze_predictability(scene_id),
                'risk_assessment': self.calculate_risk_score(scene_id),
                'safety_margins': self.analyze_safety_margins(scene_id),
                'collision_risk': self.assess_collision_risk(scene_id),
                'traffic_compliance': self.analyze_traffic_compliance(scene_id),
                'system_performance': self.detect_system_performance_issues(scene_id)
            }
            
            return analysis
        except Exception as e:
            logger.error(f"Error generating comprehensive analysis: {e}")
            return {}
    
    def analyze_all_scenes(self) -> Dict[str, Any]:
        """
        Analyze vehicle state data for all scenes.
        
        Returns:
            Dictionary containing vehicle analysis for all scenes
        """
        logger.info("Analyzing vehicle state data for all scenes...")
        
        all_scenes_analysis = {}
        scene_summaries = {
            'driving_styles': [],
            'avg_speeds': [],
            'risk_scores': [],
            'smoothness_scores': [],
            'compliance_scores': []
        }
        
        for scene_id in range(1, 7):
            try:
                scene_analysis = self.analyze_scene(scene_id)
                all_scenes_analysis[f"Scene {scene_id}"] = scene_analysis
                
                # Collect summary data for overall analysis
                if scene_analysis:
                    # Driving style
                    driving_style = scene_analysis.get('driving_style', {})
                    if driving_style:
                        scene_summaries['driving_styles'].append({
                            'scene': f"Scene {scene_id}",
                            'style': driving_style.get('style', 'unknown'),
                            'score': driving_style.get('overall_score', 0)
                        })
                    
                    # Speed data
                    velocity_summary = scene_analysis.get('velocity_summary', {})
                    if velocity_summary:
                        scene_summaries['avg_speeds'].append({
                            'scene': f"Scene {scene_id}",
                            'avg_speed': velocity_summary.get('avg_speed', 0),
                            'max_speed': velocity_summary.get('max_speed', 0)
                        })
                    
                    # Risk assessment
                    risk_assessment = scene_analysis.get('risk_assessment', {})
                    if risk_assessment:
                        scene_summaries['risk_scores'].append({
                            'scene': f"Scene {scene_id}",
                            'risk_score': risk_assessment.get('overall_risk_score', 0),
                            'risk_level': risk_assessment.get('risk_level', 'unknown')
                        })
                    
                    # Smoothness
                    smoothness = scene_analysis.get('smoothness', {})
                    if smoothness:
                        scene_summaries['smoothness_scores'].append({
                            'scene': f"Scene {scene_id}",
                            'smoothness_score': smoothness.get('overall_smoothness_score', 0)
                        })
                    
                    # Traffic compliance
                    traffic_compliance = scene_analysis.get('traffic_compliance', {})
                    if traffic_compliance:
                        scene_summaries['compliance_scores'].append({
                            'scene': f"Scene {scene_id}",
                            'compliance_level': traffic_compliance.get('compliance_level', 'unknown')
                        })
                        
            except Exception as e:
                logger.error(f"Error analyzing scene {scene_id}: {e}")
                continue
        
        return {
            'scene_analyses': all_scenes_analysis,
            'summaries': scene_summaries
        }