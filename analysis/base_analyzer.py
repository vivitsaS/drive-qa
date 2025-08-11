"""
Base Analyzer Class

Provides common functionality and interfaces for all analyzers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union, Optional
from loguru import logger

from parsers.data_loader import DataLoader


class BaseAnalyzer(ABC):
    """Base class for all analyzers with common functionality"""
    
    def __init__(self, data_loader: DataLoader = None):
        """
        Initialize base analyzer.
        
        Args:
            data_loader: DataLoader instance, creates new one if None
        """
        self.data_loader = data_loader if data_loader else DataLoader()
        self._analysis_cache: Dict[str, Any] = {}
    
    def clear_cache(self) -> None:
        """Clear the analysis cache"""
        self._analysis_cache.clear()
        logger.info("Analysis cache cleared")
    
    def get_cached_result(self, key: str) -> Optional[Any]:
        """
        Get cached analysis result.
        
        Args:
            key: Cache key
            
        Returns:
            Cached result or None if not found
        """
        return self._analysis_cache.get(key)
    
    def set_cached_result(self, key: str, result: Any) -> None:
        """
        Cache analysis result.
        
        Args:
            key: Cache key
            result: Analysis result to cache
        """
        self._analysis_cache[key] = result
    
    def get_scene_data(self, scene_id: Union[int, str]) -> Dict[str, Any]:
        """
        Get scene data with caching.
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            Scene data dictionary
        """
        cache_key = f"scene_data_{scene_id}"
        result = self.get_cached_result(cache_key)
        
        if result is None:
            result = self.data_loader.load_scene_data(scene_id)
            self.set_cached_result(cache_key, result)
        
        return result
    
    def get_all_scenes_data(self) -> Dict[int, Dict[str, Any]]:
        """
        Get data for all scenes with caching.
        
        Returns:
            Dictionary mapping scene IDs to scene data
        """
        cache_key = "all_scenes_data"
        result = self.get_cached_result(cache_key)
        
        if result is None:
            result = {}
            for scene_id in range(1, 7):
                try:
                    scene_data = self.data_loader.load_scene_data(scene_id)
                    result[scene_id] = scene_data
                except Exception as e:
                    logger.warning(f"Failed to load scene {scene_id}: {e}")
                    continue
            
            self.set_cached_result(cache_key, result)
        
        return result
    
    @abstractmethod
    def analyze_scene(self, scene_id: Union[int, str]) -> Dict[str, Any]:
        """
        Analyze a single scene.
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            Analysis results for the scene
        """
        pass
    
    @abstractmethod
    def analyze_all_scenes(self) -> Dict[str, Any]:
        """
        Analyze all scenes.
        
        Returns:
            Analysis results for all scenes
        """
        pass
    
    def validate_scene_id(self, scene_id: Union[int, str]) -> bool:
        """
        Validate if scene ID exists in the dataset.
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            True if valid, False otherwise
        """
        try:
            self.data_loader.load_scene_data(scene_id)
            return True
        except Exception:
            return False
    
    def get_available_scenes(self) -> List[int]:
        """
        Get list of available scene IDs.
        
        Returns:
            List of available scene IDs
        """
        available_scenes = []
        for scene_id in range(1, 7):
            if self.validate_scene_id(scene_id):
                available_scenes.append(scene_id)
        return available_scenes 