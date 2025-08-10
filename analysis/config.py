"""
Configuration settings for the analysis module.

Centralizes all hardcoded values, thresholds, and settings.
"""

from typing import List, Dict, Any
from pathlib import Path

# Data paths
DEFAULT_DATA_PATH = "data/concatenated_data/concatenated_data.json"
DEFAULT_OUTPUT_DIR = "reports"

# Scene configuration
SCENE_IDS = list(range(1, 7))  # Scenes 1-6
DEFAULT_SCENE_COUNT = 6

# QA types
QA_TYPES = ['perception', 'planning', 'prediction', 'behavior']

# Object types for analysis
OBJECT_TYPES = [
    'vehicle', 'pedestrian', 'bicycle', 'animal', 
    'traffic_light', 'traffic_sign', 'other'
]

# Object patterns for text analysis
OBJECT_PATTERNS = [
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

# Question pattern keywords
QUESTION_PATTERNS = {
    'what': ['what', 'what is', 'what are'],
    'where': ['where', 'where is', 'where are'],
    'when': ['when', 'when will', 'when should'],
    'how': ['how', 'how should', 'how will'],
    'why': ['why', 'why should', 'why will'],
    'status': ['status', 'state', 'condition'],
    'action': ['should', 'will', 'going to', 'planning to']
}

# Answer pattern keywords
ANSWER_PATTERNS = {
    'descriptive': ['there are', 'there is', 'many', 'several', 'various'],
    'actionable': ['should', 'will', 'need to', 'must', 'have to'],
    'conditional': ['if', 'when', 'unless', 'provided that', 'in case'],
    'temporal': ['now', 'soon', 'later', 'before', 'after', 'while'],
    'spatial': ['front', 'back', 'left', 'right', 'near', 'far', 'behind'],
    'quantitative': ['one', 'two', 'three', 'many', 'few', 'several', 'all'],
    'qualitative': ['good', 'bad', 'safe', 'dangerous', 'clear', 'obstructed']
}

# Vehicle analysis thresholds
VEHICLE_ANALYSIS_THRESHOLDS = {
    'speed_threshold': 5.0,  # m/s
    'acceleration_threshold': 2.0,  # m/s²
    'curvature_threshold': 0.015,
    'conservative_score_threshold': 0.3,
    'aggressive_score_threshold': 0.7,
    'smoothness_threshold': 0.5,
    'predictability_threshold': 0.6,
    'risk_threshold': 0.7
}

# Analysis limits
ANALYSIS_LIMITS = {
    'max_objects_displayed': 15,
    'max_objects_per_type': 10,
    'max_questions_per_scene': 1000,
    'max_keyframes_per_scene': 50
}

# Cache settings
CACHE_SETTINGS = {
    'enable_caching': True,
    'max_cache_size': 1000,
    'cache_ttl_seconds': 3600  # 1 hour
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}',
    'file': 'logs/analysis.log'
}

# Dashboard configuration
DASHBOARD_CONFIG = {
    'theme': 'light',
    'page_title': 'DriveLM QA Analysis Dashboard',
    'page_icon': '🚗',
    'layout': 'wide',
    'chart_height': 400,
    'max_charts_per_row': 2
}

# File paths and directories
PATHS = {
    'data_dir': Path('data'),
    'reports_dir': Path('reports'),
    'logs_dir': Path('logs'),
    'notebooks_dir': Path('notebooks'),
    'tests_dir': Path('tests')
}

# Ensure directories exist
for path in PATHS.values():
    if isinstance(path, Path):
        path.mkdir(exist_ok=True)

# Validation settings
VALIDATION_CONFIG = {
    'strict_mode': False,
    'validate_data_paths': True,
    'validate_scene_ids': True,
    'validate_keyframe_ids': True
}

# Performance settings
PERFORMANCE_CONFIG = {
    'batch_size': 100,
    'max_workers': 4,
    'memory_limit_mb': 1024,
    'timeout_seconds': 300
} 