"""
LLM Interface Component

Handles interactions with the language model (Gemini).
"""

from typing import Dict, Any, List, Optional
import google.generativeai as genai
from loguru import logger

from analysis.result import AnalysisResult
from config import get_config


class LLMInterface:
    """Interface for LLM interactions"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize LLM interface.
        
        Args:
            api_key: Google Gemini API key (optional, uses config if not provided)
            model_name: Model name (optional, uses config if not provided)
        """
        self.config = get_config()
        
        # Get API key
        if api_key is None:
            api_key_env = self.config.get('rag.api_key_env', 'GEMINI_API_KEY')
            api_key = self.config.get('rag.api_key') or api_key_env
        
        if not api_key:
            raise ValueError("API key not provided and GEMINI_API_KEY environment variable not set")
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Get model name
        if model_name is None:
            model_name = self.config.get('rag.model', 'gemini-1.5-flash')
        
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        
        logger.info(f"LLM interface initialized with model: {model_name}")
    
    def generate_response(self, content_parts: List[Any]) -> AnalysisResult:
        """
        Generate response from the LLM.
        
        Args:
            content_parts: List of content parts (text and images)
            
        Returns:
            AnalysisResult with LLM response
        """
        try:
            # Generate response
            response = self.model.generate_content(content_parts)
            
            return AnalysisResult.success(
                data={
                    'response_text': response.text,
                    'model_name': self.model_name,
                    'content_parts_count': len(content_parts)
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return AnalysisResult.error(
                str(e),
                metadata={'model_name': self.model_name}
            )
    
    def generate_response_with_tools(self, content_parts: List[Any], tools: List[Dict[str, Any]]) -> AnalysisResult:
        """
        Generate response from the LLM with function calling tools.
        
        Args:
            content_parts: List of content parts (text and images)
            tools: List of tool definitions for function calling
            
        Returns:
            AnalysisResult with LLM response
        """
        try:
            # Generate response with tools
            response = self.model.generate_content(content_parts, tools=tools)
            
            return AnalysisResult.success(
                data={
                    'response_text': response.text,
                    'model_name': self.model_name,
                    'content_parts_count': len(content_parts),
                    'tools_used': len(tools)
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating LLM response with tools: {e}")
            return AnalysisResult.error(
                str(e),
                metadata={'model_name': self.model_name, 'tools_count': len(tools)}
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'max_tokens': self.config.get('rag.max_tokens', 4096),
            'temperature': self.config.get('rag.temperature', 0.1)
        }
    
    def validate_content_parts(self, content_parts: List[Any]) -> AnalysisResult:
        """
        Validate content parts before sending to LLM.
        
        Args:
            content_parts: List of content parts to validate
            
        Returns:
            AnalysisResult with validation status
        """
        try:
            if not content_parts:
                return AnalysisResult.error("No content parts provided")
            
            # Check for text content
            text_parts = [part for part in content_parts if isinstance(part, str)]
            if not text_parts:
                return AnalysisResult.error("No text content found in content parts")
            
            # Check for image content
            image_parts = [part for part in content_parts if isinstance(part, dict) and 'mime_type' in part]
            
            # Validate image parts
            for i, part in enumerate(image_parts):
                if 'data' not in part:
                    return AnalysisResult.error(f"Image part {i} missing 'data' field")
                if part.get('mime_type') != 'image/png':
                    return AnalysisResult.warning(f"Image part {i} has unsupported mime_type: {part.get('mime_type')}")
            
            return AnalysisResult.success(
                data={
                    'text_parts_count': len(text_parts),
                    'image_parts_count': len(image_parts),
                    'total_parts': len(content_parts)
                }
            )
            
        except Exception as e:
            logger.error(f"Error validating content parts: {e}")
            return AnalysisResult.error(str(e)) 