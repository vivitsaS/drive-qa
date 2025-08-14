"""
Refactored RAG Agent

Uses modular components for better separation of concerns:
- DataRetriever: Handles data access
- ContextBuilder: Builds context for LLM
- LLMInterface: Handles LLM interactions
"""

from typing import Dict, Any, Union, Optional
from loguru import logger

from parsers.data_service import DataService
from analysis.result import AnalysisResult, ResultHandler
from rag.components import DataRetriever, ContextBuilder, LLMInterface
from config import get_config


class RAGAgentV2:
    """Refactored RAG Agent using modular components"""
    
    def __init__(self, data_service: Optional[DataService] = None, api_key: Optional[str] = None):
        """
        Initialize the refactored RAG agent.
        
        Args:
            data_service: Data service instance (optional, creates new one if not provided)
            api_key: Google Gemini API key (optional, uses config if not provided)
        """
        # Initialize components
        if data_service is None:
            config = get_config()
            data_service = DataService(config.get_data_path())
        
        self.data_service = data_service
        self.data_retriever = DataRetriever(data_service)
        self.context_builder = ContextBuilder(self.data_retriever)
        self.llm_interface = LLMInterface(api_key)
        
        logger.info("RAG Agent V2 initialized with modular components")
    
    def answer_question(self, scene_id: Union[int, str], keyframe_id: Union[int, str], 
                       qa_type: str, qa_serial: int) -> AnalysisResult:
        """
        Answer a question using the modular RAG system.
        
        Args:
            scene_id: Scene identifier
            keyframe_id: Keyframe identifier
            qa_type: Type of QA
            qa_serial: QA pair serial number
            
        Returns:
            AnalysisResult with the answer and metadata
        """
        try:
            # Step 1: Build context
            context_result = self.context_builder.build_context_for_qa(
                scene_id, keyframe_id, qa_type, qa_serial
            )
            
            if not context_result.status.value == 'success':
                return context_result
            
            context = context_result.data
            
            # Step 2: Build content parts for LLM
            content_parts = self.context_builder.build_content_parts(context)
            
            # Step 3: Validate content parts
            validation_result = self.llm_interface.validate_content_parts(content_parts)
            if not validation_result.status.value == 'success':
                logger.warning(f"Content validation warning: {validation_result.warning}")
            
            # Step 4: Generate LLM response
            llm_result = self.llm_interface.generate_response(content_parts)
            
            if not llm_result.status.value == 'success':
                return llm_result
            
            # Step 5: Build final result
            response_data = llm_result.data
            model_answer = response_data.get('response_text', '')
            
            return AnalysisResult.success(
                data={
                    'question': context.get('question', ''),
                    'model_answer': model_answer,
                    'ground_truth_answer': context.get('ground_truth', ''),
                    'context_metadata': context.get('metadata', {}),
                    'llm_metadata': response_data
                },
                metadata={
                    'scene_id': scene_id,
                    'keyframe_id': keyframe_id,
                    'qa_type': qa_type,
                    'qa_serial': qa_serial,
                    'model_name': response_data.get('model_name'),
                    'content_parts_count': response_data.get('content_parts_count')
                }
            )
            
        except Exception as e:
            logger.error(f"Error in answer_question: {e}")
            return AnalysisResult.error(
                str(e),
                metadata={
                    'scene_id': scene_id,
                    'keyframe_id': keyframe_id,
                    'qa_type': qa_type,
                    'qa_serial': qa_serial
                }
            )
    
    def get_context_info(self, scene_id: Union[int, str], keyframe_id: Union[int, str]) -> AnalysisResult:
        """
        Get information about available context for a keyframe.
        
        Args:
            scene_id: Scene identifier
            keyframe_id: Keyframe identifier
            
        Returns:
            AnalysisResult with context information
        """
        try:
            # Get various data sources
            context_result = self.data_retriever.get_context_for_keyframe(scene_id, keyframe_id)
            vehicle_result = self.data_retriever.get_vehicle_data_upto_keyframe(scene_id, keyframe_id)
            sensor_result = self.data_retriever.get_sensor_data_for_keyframe(scene_id, keyframe_id)
            images_result = self.data_retriever.get_annotated_images(scene_id, keyframe_id)
            keyframe_result = self.data_retriever.get_keyframe_info(scene_id, keyframe_id)
            
            # Build context info
            context_info = {
                'scene_id': scene_id,
                'keyframe_id': keyframe_id,
                'data_sources': {
                    'context_data': context_result.status.value == 'success',
                    'vehicle_data': vehicle_result.status.value == 'success',
                    'sensor_data': sensor_result.status.value == 'success',
                    'annotated_images': images_result.status.value == 'success',
                    'keyframe_info': keyframe_result.status.value == 'success'
                },
                'available_qa_types': self._get_available_qa_types(scene_id, keyframe_id)
            }
            
            return AnalysisResult.success(data=context_info)
            
        except Exception as e:
            logger.error(f"Error getting context info: {e}")
            return AnalysisResult.error(str(e))
    
    def _get_available_qa_types(self, scene_id: Union[int, str], keyframe_id: Union[int, str]) -> Dict[str, int]:
        """
        Get available QA types and their counts for a keyframe.
        
        Args:
            scene_id: Scene identifier
            keyframe_id: Keyframe identifier
            
        Returns:
            Dictionary mapping QA types to their counts
        """
        try:
            qa_pairs = self.data_service.get_qa_pairs(scene_id, keyframe_id)
            
            if isinstance(qa_pairs, dict):
                # If qa_pairs is a dict (all keyframes), find the specific keyframe
                keyframe_token = self.data_service.data_loader._assign_keyframe_token(scene_id, keyframe_id)
                if keyframe_token in qa_pairs:
                    qa_pairs = qa_pairs[keyframe_token]
                else:
                    return {}
            
            # Count QA types
            qa_counts = {}
            for qa_type, pairs in qa_pairs.items():
                if isinstance(pairs, list):
                    qa_counts[qa_type] = len(pairs)
                else:
                    qa_counts[qa_type] = 0
            
            return qa_counts
            
        except Exception as e:
            logger.error(f"Error getting QA types: {e}")
            return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the LLM model.
        
        Returns:
            Dictionary with model information
        """
        return self.llm_interface.get_model_info()
    
    def validate_inputs(self, scene_id: Union[int, str], keyframe_id: Union[int, str], 
                       qa_type: str, qa_serial: int) -> AnalysisResult:
        """
        Validate input parameters before processing.
        
        Args:
            scene_id: Scene identifier
            keyframe_id: Keyframe identifier
            qa_type: Type of QA
            qa_serial: QA pair serial number
            
        Returns:
            AnalysisResult with validation status
        """
        try:
            # Validate scene ID
            available_scenes = self.data_service.get_available_scenes()
            scene_validation = ResultHandler.validate_scene_id(scene_id, available_scenes)
            if not scene_validation.status.value == 'success':
                return scene_validation
            
            # Validate keyframe ID (basic validation)
            try:
                int(keyframe_id)
            except ValueError:
                return AnalysisResult.error(f"Invalid keyframe_id: {keyframe_id}")
            
            # Validate QA type
            valid_qa_types = ['perception', 'planning', 'prediction', 'behavior']
            if qa_type not in valid_qa_types:
                return AnalysisResult.error(
                    f"Invalid qa_type: {qa_type}. Valid types: {valid_qa_types}"
                )
            
            # Validate QA serial
            try:
                qa_serial_int = int(qa_serial)
                if qa_serial_int < 1:
                    return AnalysisResult.error(f"qa_serial must be >= 1, got: {qa_serial}")
            except ValueError:
                return AnalysisResult.error(f"Invalid qa_serial: {qa_serial}")
            
            return AnalysisResult.success()
            
        except Exception as e:
            logger.error(f"Error validating inputs: {e}")
            return AnalysisResult.error(str(e)) 