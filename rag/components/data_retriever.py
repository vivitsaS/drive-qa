"""
Data Retriever Component

Handles all data retrieval operations for the RAG system.
Uses the unified DataService for consistent data access.
"""

from typing import Dict, Any, Optional, Union
import base64
from loguru import logger

from parsers.data_service import DataService
from analysis.result import AnalysisResult, ResultStatus


class DataRetriever:
    """Handles data retrieval operations for RAG system"""
    
    def __init__(self, data_service: DataService):
        """
        Initialize data retriever.
        
        Args:
            data_service: Unified data service instance
        """
        self.data_service = data_service
    
    def get_context_for_keyframe(self, scene_id: Union[int, str], keyframe_id: Union[int, str]) -> AnalysisResult:
        """
        Get context data for a specific keyframe.
        
        Args:
            scene_id: Scene identifier
            keyframe_id: Keyframe identifier
            
        Returns:
            AnalysisResult with context data
        """
        try:
            context_data = self.data_service.get_context_for_keyframe(scene_id, keyframe_id)
            
            if not context_data:
                return AnalysisResult.error(
                    f"No context data found for scene {scene_id}, keyframe {keyframe_id}",
                    metadata={'scene_id': scene_id, 'keyframe_id': keyframe_id}
                )
            
            return AnalysisResult.success(
                data={
                    "scene_id": scene_id,
                    "keyframe_id": keyframe_id,
                    "samples_count": len(context_data.get("samples", {})),
                    "keyframes_count": len(context_data.get("key_frames", {}))
                },
                metadata={'context_data': context_data}
            )
            
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return AnalysisResult.error(
                str(e),
                metadata={'scene_id': scene_id, 'keyframe_id': keyframe_id}
            )
    
    def get_vehicle_data_upto_keyframe(self, scene_id: Union[int, str], keyframe_id: Union[int, str]) -> AnalysisResult:
        """
        Get vehicle movement data up to a specific keyframe.
        
        Args:
            scene_id: Scene identifier
            keyframe_id: Keyframe identifier
            
        Returns:
            AnalysisResult with vehicle data
        """
        try:
            vehicle_data = self.data_service.get_vehicle_data_upto_keyframe(scene_id, keyframe_id)
            
            if not vehicle_data:
                return AnalysisResult.error(
                    f"No vehicle data found for scene {scene_id}, keyframe {keyframe_id}",
                    metadata={'scene_id': scene_id, 'keyframe_id': keyframe_id}
                )
            
            return AnalysisResult.success(data=vehicle_data)
            
        except Exception as e:
            logger.error(f"Error getting vehicle data: {e}")
            return AnalysisResult.error(
                str(e),
                metadata={'scene_id': scene_id, 'keyframe_id': keyframe_id}
            )
    
    def get_sensor_data_for_keyframe(self, scene_id: Union[int, str], keyframe_id: Union[int, str]) -> AnalysisResult:
        """
        Get sensor data for a specific keyframe.
        
        Args:
            scene_id: Scene identifier
            keyframe_id: Keyframe identifier
            
        Returns:
            AnalysisResult with sensor data
        """
        try:
            sensor_data = self.data_service.get_sensor_data_for_keyframe(scene_id, keyframe_id)
            
            if not sensor_data:
                return AnalysisResult.error(
                    f"No sensor data found for scene {scene_id}, keyframe {keyframe_id}",
                    metadata={'scene_id': scene_id, 'keyframe_id': keyframe_id}
                )
            
            return AnalysisResult.success(data=sensor_data)
            
        except Exception as e:
            logger.error(f"Error getting sensor data: {e}")
            return AnalysisResult.error(
                str(e),
                metadata={'scene_id': scene_id, 'keyframe_id': keyframe_id}
            )
    
    def get_qa_pair(self, scene_id: Union[int, str], keyframe_id: Union[int, str], 
                   qa_type: str, qa_serial: int) -> AnalysisResult:
        """
        Get a specific QA pair.
        
        Args:
            scene_id: Scene identifier
            keyframe_id: Keyframe identifier
            qa_type: Type of QA
            qa_serial: QA pair serial number
            
        Returns:
            AnalysisResult with QA pair data
        """
        try:
            qa_pair = self.data_service.get_qa_pair(scene_id, keyframe_id, qa_type, qa_serial)
            
            if not qa_pair:
                return AnalysisResult.error(
                    f"QA pair not found for type={qa_type}, serial={qa_serial}",
                    metadata={
                        'scene_id': scene_id,
                        'keyframe_id': keyframe_id,
                        'qa_type': qa_type,
                        'qa_serial': qa_serial
                    }
                )
            
            return AnalysisResult.success(data=qa_pair)
            
        except Exception as e:
            logger.error(f"Error getting QA pair: {e}")
            return AnalysisResult.error(
                str(e),
                metadata={
                    'scene_id': scene_id,
                    'keyframe_id': keyframe_id,
                    'qa_type': qa_type,
                    'qa_serial': qa_serial
                }
            )
    
    def get_keyframe_info(self, scene_id: Union[int, str], keyframe_id: Union[int, str]) -> AnalysisResult:
        """
        Get keyframe information including key objects.
        
        Args:
            scene_id: Scene identifier
            keyframe_id: Keyframe identifier
            
        Returns:
            AnalysisResult with keyframe info
        """
        try:
            keyframe_data = self.data_service.get_keyframe_data(scene_id, keyframe_id)
            
            if not keyframe_data:
                return AnalysisResult.error(
                    f"No keyframe data found for scene {scene_id}, keyframe {keyframe_id}",
                    metadata={'scene_id': scene_id, 'keyframe_id': keyframe_id}
                )
            
            # Extract key objects from DriveLM data
            drivelm_data = keyframe_data.get('DriveLM_data', {})
            key_objects = drivelm_data.get('key_object_infos', {})
            
            return AnalysisResult.success(
                data={
                    'scene_info': keyframe_data.get('scene_info', {}),
                    'key_objects': key_objects,
                    'keyframe_token': keyframe_data.get('scene_info', {}).get('keyframe_token')
                }
            )
            
        except Exception as e:
            logger.error(f"Error getting keyframe info: {e}")
            return AnalysisResult.error(
                str(e),
                metadata={'scene_id': scene_id, 'keyframe_id': keyframe_id}
            )
    
    def get_annotated_images(self, scene_id: Union[int, str], keyframe_id: Union[int, str]) -> AnalysisResult:
        """
        Get annotated images for a keyframe.
        
        Args:
            scene_id: Scene identifier
            keyframe_id: Keyframe identifier
            
        Returns:
            AnalysisResult with image data
        """
        try:
            # For now, we'll use the existing ContextRetriever for image generation
            # This could be refactored to use a dedicated image service in the future
            from rag.retrieval.context_retriever import ContextRetriever
            
            context_retriever = ContextRetriever(scene_id, keyframe_id)
            image_bytes = context_retriever.get_annotated_images()
            
            if not image_bytes:
                return AnalysisResult.error(
                    f"No annotated images found for scene {scene_id}, keyframe {keyframe_id}",
                    metadata={'scene_id': scene_id, 'keyframe_id': keyframe_id}
                )
            
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            return AnalysisResult.success(
                data={
                    "image_base64": image_base64,
                    "format": "png",
                    "description": "Annotated images showing detected objects with bounding boxes"
                }
            )
            
        except Exception as e:
            logger.error(f"Error getting annotated images: {e}")
            return AnalysisResult.error(
                str(e),
                metadata={'scene_id': scene_id, 'keyframe_id': keyframe_id}
            ) 