"""
Model Loader for RAG Pipeline

Handles loading and management of LLaVA and Llama 3 models.
"""

from typing import Optional, Dict, Any
from loguru import logger

from .llava_model import LLaVAModel
from .llama_model import LlamaModel
from .api_models import MistralAPIModel, OpenAIGPTVisionModel, AnthropicClaudeVisionModel


class ModelLoader:
    """Central model loading and management system"""
    
    def __init__(self, device: str = "auto", quantization: Optional[str] = None, use_api: bool = True):
        """
        Initialize model loader.
        
        Args:
            device: Device to load models on ("cuda", "cpu", "auto")
            quantization: Quantization type ("4bit", "8bit", None)
            use_api: Whether to use API models for faster inference
        """
        self.device = device
        self.quantization = quantization
        self.use_api = use_api
        self._llava_model: Optional[LLaVAModel] = None
        self._llama_model: Optional[LlamaModel] = None
        self._api_llama_model: Optional[MistralAPIModel] = None
        self._api_vision_model: Optional[OpenAIGPTVisionModel] = None
        
    def load_llava(self, model_name: str = "gpt-4-vision-preview"):
        """
        Load vision model for multi-modal reasoning.
        
        Args:
            model_name: Model identifier (API model name or HF model)
            
        Returns:
            Loaded vision model instance
        """
        if self.use_api:
            if self._api_vision_model is None:
                logger.info(f"Loading API vision model: {model_name}")
                if "gpt" in model_name.lower():
                    self._api_vision_model = OpenAIGPTVisionModel(model_name=model_name)
                elif "claude" in model_name.lower():
                    self._api_vision_model = AnthropicClaudeVisionModel(model_name=model_name)
                else:
                    # Default to OpenAI GPT-4V
                    self._api_vision_model = OpenAIGPTVisionModel()
            return self._api_vision_model
        else:
            if self._llava_model is None:
                logger.info(f"Loading local LLaVA model: {model_name}")
                self._llava_model = LLaVAModel(model_name, self.device, self.quantization)
            return self._llava_model
        
    def load_llama(self, model_name: str = "mistral-large-latest"):
        """
        Load text model for reasoning.
        
        Args:
            model_name: Model identifier (API model name or HF model)
            
        Returns:
            Loaded text model instance
        """
        if self.use_api:
            if self._api_llama_model is None:
                logger.info(f"Loading API text model: {model_name}")
                self._api_llama_model = MistralAPIModel(model_name=model_name)
            return self._api_llama_model
        else:
            if self._llama_model is None:
                logger.info(f"Loading local text model: {model_name}")
                # Fallback to a smaller model for local inference
                local_model = "microsoft/DialoGPT-medium" if "mistral" in model_name.lower() else model_name
                self._llama_model = LlamaModel(local_model, self.device, self.quantization)
            return self._llama_model
        
    def unload_models(self):
        """Unload all models to free memory"""
        if self._llava_model:
            self._llava_model.unload()
            self._llava_model = None
            
        if self._llama_model:
            self._llama_model.unload()
            self._llama_model = None
            
        if self._api_llama_model:
            self._api_llama_model.unload()
            self._api_llama_model = None
            
        if self._api_vision_model:
            self._api_vision_model.unload()
            self._api_vision_model = None
            
        logger.info("All models unloaded")
        
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get current memory usage of loaded models.
        
        Returns:
            Dictionary with memory usage statistics
        """
        usage = {
            "llava_loaded": self._llava_model is not None,
            "llama_loaded": self._llama_model is not None,
            "total_memory_mb": 0
        }
        
        # TODO: Implement memory tracking
        return usage