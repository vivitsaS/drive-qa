"""
Predictor Analyzer

Analyzes which data fields are the best predictors for different QA types:
- Feature importance analysis for each QA type
- Correlation analysis between data fields and QA types
- Identification of key indicators for each question category
"""

import json
from typing import Dict, List, Any, Union, Tuple
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger

from .data_loader import DataLoader


class PredictorAnalyzer:
    """Analyze which data fields best predict different QA types"""
    
    def __init__(self, data_loader: DataLoader):
        """Initialize the predictor analyzer"""
        self.data_loader = data_loader
        self.qa_types = ['perception', 'planning', 'prediction', 'behavior']
        
    def analyze_qa_type_predictors(self) -> Dict[str, Any]:
        """
        Analyze which data fields are the best predictors for each QA type.
        
        Returns:
            Dictionary containing predictor analysis for each QA type
        """
        logger.info("Analyzing predictors for QA types...")
        
        # Collect all data points with their features and QA types
        all_data_points = self._collect_data_points()
        
        # Analyze predictors for each QA type
        predictor_results = {}
        for qa_type in self.qa_types:
            logger.info(f"Analyzing predictors for {qa_type} questions...")
            predictors = self._analyze_predictors_for_qa_type(all_data_points, qa_type)
            predictor_results[qa_type] = predictors
        
        return predictor_results
    
    def _collect_data_points(self) -> List[Dict[str, Any]]:
        """Collect all data points with their features and QA types"""
        data_points = []
        
        for scene_id in range(1, 7):
            scene_data = self.data_loader.load_scene_data(scene_id)
            
            # Get QA data for all keyframes in this scene
            qa_data = self.data_loader.extract_questions_from_keyframe(scene_id, 0)  # 0 for all keyframes
            
            # Get keyframes from scene data
            scene_keyframes = scene_data.get('key_frames', {})
            
            # For each keyframe, create a data point
            for keyframe_token, keyframe_data in scene_keyframes.items():
                # Create synthetic features for this keyframe
                features = self._extract_keyframe_features(keyframe_data, scene_data)
                
                # Count QA types for this keyframe
                qa_counts = {qa_type: 0 for qa_type in self.qa_types}
                
                if keyframe_token in qa_data:
                    keyframe_qa = qa_data[keyframe_token]
                    for qa_type in self.qa_types:
                        if qa_type in keyframe_qa and keyframe_qa[qa_type]:
                            qa_counts[qa_type] += len(keyframe_qa[qa_type])
                
                # Create data point for each QA type
                for qa_type in self.qa_types:
                    data_point = features.copy()
                    data_point['qa_type'] = qa_type
                    data_point['has_qa'] = qa_counts[qa_type] > 0
                    data_point['qa_count'] = qa_counts[qa_type]
                    data_points.append(data_point)
                
                # Also add data points for keyframes without QA data (to create variation)
                if keyframe_token not in qa_data:
                    for qa_type in self.qa_types:
                        data_point = features.copy()
                        data_point['qa_type'] = qa_type
                        data_point['has_qa'] = False
                        data_point['qa_count'] = 0
                        data_points.append(data_point)
        
        return data_points
    
    def _extract_keyframe_features(self, keyframe_data: Dict[str, Any], scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from a keyframe"""
        features = {}
        
        # Object detection features from keyframe
        key_object_infos = keyframe_data.get('key_object_infos', {})
        total_objects = len(key_object_infos)
        features['total_objects'] = total_objects
        
        # Count object types
        object_types = {}
        for obj_id, obj_info in key_object_infos.items():
            category = obj_info.get('Category', 'unknown')
            object_types[category] = object_types.get(category, 0) + 1
        
        features['unique_object_types'] = len(object_types)
        features['object_density'] = total_objects
        
        # Most common object types
        for obj_type, count in sorted(object_types.items(), key=lambda x: x[1], reverse=True)[:5]:
            features[f'count_{obj_type.replace(" ", "_").lower()}'] = count
        
        # Scene context features
        scene_description = scene_data.get('scene_description', '').lower()
        features['is_intersection'] = any(word in scene_description for word in ['intersection', 'crossing'])
        features['is_parking'] = any(word in scene_description for word in ['parking', 'parked'])
        features['is_highway'] = any(word in scene_description for word in ['highway', 'freeway'])
        
        # Synthetic sensor features (based on scene context)
        features['active_cameras'] = 6  # Assume all cameras are active
        features['active_radars'] = 5   # Assume all radars are active
        features['active_lidars'] = 1   # Assume lidar is active
        features['camera_coverage'] = 1.0
        features['radar_coverage'] = 1.0
        features['lidar_coverage'] = 1.0
        features['total_sensors'] = 12
        features['sensor_fusion_score'] = 1.0
        
        # Synthetic vehicle state features
        features['speed'] = 5.0  # Assume moderate speed
        features['acceleration'] = 0.0  # Assume constant speed
        features['curvature'] = 0.0  # Assume straight driving
        features['angular_velocity'] = 0.0  # Assume no turning
        
        # Scene and temporal features
        features['scene_id'] = scene_data.get('scene_id', 0)
        features['keyframe_id'] = len(str(keyframe_data)) % 1000  # Synthetic keyframe ID
        
        return features
    
    def _analyze_predictors_for_qa_type(self, data_points: List[Dict[str, Any]], qa_type: str) -> Dict[str, Any]:
        """Analyze predictors for a specific QA type"""
        # Filter data points for this QA type
        qa_data = [dp for dp in data_points if dp['qa_type'] == qa_type]
        
        if not qa_data:
            return {'error': f'No data found for {qa_type} questions'}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(qa_data)
        
        # Separate features and target
        feature_columns = [col for col in df.columns if col not in ['qa_type', 'has_qa', 'qa_count']]
        
        # Analyze binary prediction (has QA or not)
        binary_correlations = self._analyze_binary_correlations(df, feature_columns, 'has_qa')
        
        # Analyze count prediction (number of QAs)
        count_correlations = self._analyze_count_correlations(df, feature_columns, 'qa_count')
        
        # Rank features by importance
        feature_importance = self._rank_feature_importance(binary_correlations, count_correlations)
        
        # Identify key indicators
        key_indicators = self._identify_key_indicators(df, feature_columns, 'has_qa')
        
        return {
            'binary_correlations': binary_correlations,
            'count_correlations': count_correlations,
            'feature_importance': feature_importance,
            'key_indicators': key_indicators,
            'data_summary': {
                'total_samples': len(df),
                'samples_with_qa': df['has_qa'].sum(),
                'avg_qa_count': df['qa_count'].mean(),
                'max_qa_count': df['qa_count'].max()
            }
        }
    
    def _analyze_binary_correlations(self, df: pd.DataFrame, features: List[str], target: str) -> Dict[str, float]:
        """Analyze correlations between features and binary target"""
        correlations = {}
        
        for feature in features:
            try:
                if df[feature].dtype in ['int64', 'float64']:
                    # Check if feature has variation
                    if df[feature].std() == 0:
                        # Constant feature - no correlation possible
                        correlations[feature] = {
                            'correlation': 0.0,
                            'p_value': 1.0,
                            'abs_correlation': 0.0
                        }
                    else:
                        # Point-biserial correlation for continuous features
                        correlation, p_value = stats.pointbiserialr(df[target], df[feature])
                        correlations[feature] = {
                            'correlation': correlation if not np.isnan(correlation) else 0.0,
                            'p_value': p_value if not np.isnan(p_value) else 1.0,
                            'abs_correlation': abs(correlation) if not np.isnan(correlation) else 0.0
                        }
                else:
                    # Chi-square test for categorical features
                    contingency_table = pd.crosstab(df[feature], df[target])
                    if contingency_table.size > 0 and contingency_table.values.min() > 0:
                        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                        correlations[feature] = {
                            'chi2': chi2 if not np.isnan(chi2) else 0.0,
                            'p_value': p_value if not np.isnan(p_value) else 1.0,
                            'abs_correlation': np.sqrt(chi2 / (chi2 + len(df))) if not np.isnan(chi2) else 0.0
                        }
                    else:
                        correlations[feature] = {
                            'chi2': 0.0,
                            'p_value': 1.0,
                            'abs_correlation': 0.0
                        }
            except Exception as e:
                # Handle any other errors
                correlations[feature] = {
                    'correlation': 0.0,
                    'p_value': 1.0,
                    'abs_correlation': 0.0
                }
        
        return correlations
    
    def _analyze_count_correlations(self, df: pd.DataFrame, features: List[str], target: str) -> Dict[str, float]:
        """Analyze correlations between features and count target"""
        correlations = {}
        
        for feature in features:
            try:
                if df[feature].dtype in ['int64', 'float64']:
                    # Check if feature has variation
                    if df[feature].std() == 0:
                        # Constant feature - no correlation possible
                        correlations[feature] = {
                            'correlation': 0.0,
                            'p_value': 1.0,
                            'abs_correlation': 0.0
                        }
                    else:
                        # Pearson correlation for continuous features
                        correlation, p_value = stats.pearsonr(df[target], df[feature])
                        correlations[feature] = {
                            'correlation': correlation if not np.isnan(correlation) else 0.0,
                            'p_value': p_value if not np.isnan(p_value) else 1.0,
                            'abs_correlation': abs(correlation) if not np.isnan(correlation) else 0.0
                        }
            except Exception as e:
                # Handle any other errors
                correlations[feature] = {
                    'correlation': 0.0,
                    'p_value': 1.0,
                    'abs_correlation': 0.0
                }
        
        return correlations
    
    def _rank_feature_importance(self, binary_corr: Dict[str, Any], count_corr: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank features by their importance across both analyses"""
        feature_scores = {}
        
        # Combine scores from both analyses
        for feature in set(binary_corr.keys()) | set(count_corr.keys()):
            binary_score = binary_corr.get(feature, {}).get('abs_correlation', 0)
            count_score = count_corr.get(feature, {}).get('abs_correlation', 0)
            
            # Weighted average (binary prediction might be more important)
            combined_score = (0.7 * binary_score) + (0.3 * count_score)
            
            feature_scores[feature] = {
                'feature': feature,
                'binary_correlation': binary_score,
                'count_correlation': count_score,
                'combined_score': combined_score
            }
        
        # Sort by combined score
        ranked_features = sorted(
            feature_scores.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )
        
        return ranked_features
    
    def _identify_key_indicators(self, df: pd.DataFrame, features: List[str], target: str) -> Dict[str, Any]:
        """Identify the strongest single indicators"""
        key_indicators = {}
        
        # Find features with highest correlation
        correlations = self._analyze_binary_correlations(df, features, target)
        
        # Top 5 strongest predictors
        top_predictors = sorted(
            correlations.items(),
            key=lambda x: x[1]['abs_correlation'],
            reverse=True
        )[:5]
        
        key_indicators['top_predictors'] = [
            {
                'feature': feature,
                'correlation': corr['correlation'],
                'p_value': corr['p_value'],
                'strength': 'strong' if corr['abs_correlation'] > 0.5 else 'moderate' if corr['abs_correlation'] > 0.3 else 'weak'
            }
            for feature, corr in top_predictors
        ]
        
        # Threshold-based indicators
        key_indicators['threshold_indicators'] = self._find_threshold_indicators(df, features, target)
        
        return key_indicators
    
    def _find_threshold_indicators(self, df: pd.DataFrame, features: List[str], target: str) -> List[Dict[str, Any]]:
        """Find features that are good indicators when above/below certain thresholds"""
        threshold_indicators = []
        
        for feature in features:
            if df[feature].dtype in ['int64', 'float64']:
                # Find optimal threshold
                thresholds = np.percentile(df[feature], [25, 50, 75])
                
                for threshold in thresholds:
                    # Above threshold
                    above_threshold = df[df[feature] > threshold][target].mean()
                    below_threshold = df[df[feature] <= threshold][target].mean()
                    
                    # Below threshold
                    if above_threshold > below_threshold and above_threshold > 0.6:
                        threshold_indicators.append({
                            'feature': feature,
                            'condition': f'{feature} > {threshold:.2f}',
                            'qa_probability': above_threshold,
                            'baseline_probability': below_threshold,
                            'improvement': above_threshold - below_threshold
                        })
                    elif below_threshold > above_threshold and below_threshold > 0.6:
                        threshold_indicators.append({
                            'feature': feature,
                            'condition': f'{feature} <= {threshold:.2f}',
                            'qa_probability': below_threshold,
                            'baseline_probability': above_threshold,
                            'improvement': below_threshold - above_threshold
                        })
        
        # Sort by improvement
        threshold_indicators.sort(key=lambda x: x['improvement'], reverse=True)
        return threshold_indicators[:10]  # Top 10 