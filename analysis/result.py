"""
Standardized Result Format

Provides consistent result and error handling across all analysis operations.
"""

from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
from loguru import logger


class ResultStatus(Enum):
    """Status of analysis result"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PARTIAL = "partial"


@dataclass
class AnalysisResult:
    """Standardized analysis result format"""
    
    status: ResultStatus
    data: Optional[Any] = None
    error: Optional[str] = None
    warning: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate result after initialization"""
        if self.status == ResultStatus.ERROR and not self.error:
            logger.warning("Error status set but no error message provided")
        if self.status == ResultStatus.WARNING and not self.warning:
            logger.warning("Warning status set but no warning message provided")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format"""
        result = {
            'status': self.status.value,
            'success': self.status == ResultStatus.SUCCESS
        }
        
        if self.data is not None:
            result['data'] = self.data
        if self.error is not None:
            result['error'] = self.error
        if self.warning is not None:
            result['warning'] = self.warning
        if self.metadata is not None:
            result['metadata'] = self.metadata
        
        return result
    
    def to_json(self) -> str:
        """Convert result to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def success(cls, data: Any = None, metadata: Optional[Dict[str, Any]] = None) -> 'AnalysisResult':
        """Create a successful result"""
        return cls(
            status=ResultStatus.SUCCESS,
            data=data,
            metadata=metadata
        )
    
    @classmethod
    def error(cls, error: str, metadata: Optional[Dict[str, Any]] = None) -> 'AnalysisResult':
        """Create an error result"""
        return cls(
            status=ResultStatus.ERROR,
            error=error,
            metadata=metadata
        )
    
    @classmethod
    def warning(cls, warning: str, data: Any = None, metadata: Optional[Dict[str, Any]] = None) -> 'AnalysisResult':
        """Create a warning result"""
        return cls(
            status=ResultStatus.WARNING,
            data=data,
            warning=warning,
            metadata=metadata
        )
    
    @classmethod
    def partial(cls, data: Any, warning: str, metadata: Optional[Dict[str, Any]] = None) -> 'AnalysisResult':
        """Create a partial result with warning"""
        return cls(
            status=ResultStatus.PARTIAL,
            data=data,
            warning=warning,
            metadata=metadata
        )


class ResultHandler:
    """Utility class for handling analysis results"""
    
    @staticmethod
    def handle_exception(func):
        """Decorator to handle exceptions and return standardized results"""
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                if isinstance(result, AnalysisResult):
                    return result
                else:
                    return AnalysisResult.success(result)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                return AnalysisResult.error(str(e))
        return wrapper
    
    @staticmethod
    def validate_scene_id(scene_id: Union[int, str], available_scenes: list) -> AnalysisResult:
        """Validate scene ID and return appropriate result"""
        try:
            scene_id_int = int(scene_id)
            if scene_id_int not in available_scenes:
                return AnalysisResult.error(
                    f"Scene ID {scene_id} not found. Available scenes: {available_scenes}",
                    metadata={'scene_id': scene_id, 'available_scenes': available_scenes}
                )
            return AnalysisResult.success()
        except ValueError:
            return AnalysisResult.error(
                f"Invalid scene ID format: {scene_id}. Expected integer.",
                metadata={'scene_id': scene_id}
            )
    
    @staticmethod
    def validate_keyframe_id(scene_id: Union[int, str], keyframe_id: Union[int, str], 
                           available_keyframes: list) -> AnalysisResult:
        """Validate keyframe ID and return appropriate result"""
        try:
            keyframe_id_int = int(keyframe_id)
            if keyframe_id_int not in available_keyframes:
                return AnalysisResult.error(
                    f"Keyframe ID {keyframe_id} not found in scene {scene_id}. "
                    f"Available keyframes: {available_keyframes}",
                    metadata={
                        'scene_id': scene_id,
                        'keyframe_id': keyframe_id,
                        'available_keyframes': available_keyframes
                    }
                )
            return AnalysisResult.success()
        except ValueError:
            return AnalysisResult.error(
                f"Invalid keyframe ID format: {keyframe_id}. Expected integer.",
                metadata={'scene_id': scene_id, 'keyframe_id': keyframe_id}
            )
    
    @staticmethod
    def aggregate_results(results: list) -> AnalysisResult:
        """Aggregate multiple results into a single result"""
        if not results:
            return AnalysisResult.error("No results to aggregate")
        
        # Check if all results are successful
        all_successful = all(r.status == ResultStatus.SUCCESS for r in results)
        all_errors = all(r.status == ResultStatus.ERROR for r in results)
        
        if all_successful:
            # Combine all data
            combined_data = [r.data for r in results if r.data is not None]
            combined_metadata = {}
            for r in results:
                if r.metadata:
                    combined_metadata.update(r.metadata)
            
            return AnalysisResult.success(combined_data, combined_metadata)
        
        elif all_errors:
            # Combine all errors
            errors = [r.error for r in results if r.error]
            combined_error = "; ".join(errors)
            return AnalysisResult.error(combined_error)
        
        else:
            # Mixed results - return partial success
            successful_data = [r.data for r in results if r.status == ResultStatus.SUCCESS and r.data is not None]
            errors = [r.error for r in results if r.status == ResultStatus.ERROR and r.error]
            
            warning = f"Partial success. Errors: {'; '.join(errors)}" if errors else "Partial success"
            return AnalysisResult.partial(successful_data, warning)
    
    @staticmethod
    def log_result(result: AnalysisResult, operation: str, **kwargs):
        """Log result with appropriate level"""
        metadata_str = f" | Metadata: {result.metadata}" if result.metadata else ""
        
        if result.status == ResultStatus.SUCCESS:
            logger.info(f"{operation} completed successfully{metadata_str}")
        elif result.status == ResultStatus.ERROR:
            logger.error(f"{operation} failed: {result.error}{metadata_str}")
        elif result.status == ResultStatus.WARNING:
            logger.warning(f"{operation} completed with warning: {result.warning}{metadata_str}")
        elif result.status == ResultStatus.PARTIAL:
            logger.warning(f"{operation} completed partially: {result.warning}{metadata_str}") 