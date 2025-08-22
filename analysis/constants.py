"""
Constants for the analysis module.

This module imports constants from the config module for backward compatibility.
"""

from .config import (
    OBJECT_TYPES, 
    QA_TYPES, 
    SCENE_IDS, 
    VEHICLE_ANALYSIS_THRESHOLDS,
    SCENE_TOKEN_MAPPINGS,
    KEYFRAME_TOKEN_MAPPINGS
)

# Re-export for backward compatibility
__all__ = [
    'OBJECT_TYPES',
    'QA_TYPES', 
    'SCENE_IDS',
    'VEHICLE_ANALYSIS_THRESHOLDS',
    'SCENE_TOKEN_MAPPINGS',
    'KEYFRAME_TOKEN_MAPPINGS'
]