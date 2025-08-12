"""
API-based model implementations for fast inference
"""

import os
import base64
import json
from typing import Optional, Union, Dict, Any
from loguru import logger
from PIL import Image
import io
import requests


class MistralAPIModel:
    """Mistral API model for fast text inference"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "mistral-large-latest"):
        """
        Initialize Mistral API client.
        
        Args:
            api_key: Mistral API key (or set MISTRAL_API_KEY env var)
            model_name: Model to use (mistral-large-latest, mistral-medium, etc.)
        """
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.model_name = model_name
        self.base_url = "https://api.mistral.ai/v1/chat/completions"
        
        if not self.api_key:
            logger.warning("No Mistral API key provided. Set MISTRAL_API_KEY environment variable.")
    
    def generate_response(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """Generate response using Mistral API"""
        try:
            if not self.api_key:
                return "Error: No Mistral API key provided"
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"Mistral API error: {e}")
            return f"Error: {e}"
    
    def analyze_structured_data(self, data: dict, question: str) -> str:
        """Analyze structured driving data"""
        system_prompt = """You are an expert in autonomous driving data analysis. 
        Analyze the provided vehicle and sensor data to answer questions accurately and concisely.
        Focus on the specific question asked and provide clear, factual answers."""
        
        prompt = f"""Given this driving scene data:

Vehicle State: {data.get('vehicle_data', {})}
Detected Objects: {len(data.get('sensor_data', {}).get('objects', []))} objects
Object Details: {data.get('sensor_data', {}).get('objects', [])[:5]}  # First 5 objects

Question: {question}

Please provide a clear, specific answer based on the data."""
        
        return self.generate_response(prompt, system_prompt, max_tokens=150)
    
    def is_loaded(self) -> bool:
        """Check if API is accessible"""
        return bool(self.api_key)
    
    def unload(self):
        """No-op for API models"""
        pass
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """API models don't use local memory"""
        return {"type": "api", "memory_mb": 0}


class OpenAIGPTVisionModel:
    """OpenAI GPT-4V for multimodal inference"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4-vision-preview"):
        """
        Initialize OpenAI API client.
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model_name: Model to use (gpt-4-vision-preview, gpt-4o, etc.)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.base_url = "https://api.openai.com/v1/chat/completions"
        
        if not self.api_key:
            logger.warning("No OpenAI API key provided. Set OPENAI_API_KEY environment variable.")
    
    def generate_response(
        self, 
        prompt: str, 
        image: Optional[Union[Image.Image, str, bytes]] = None,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """Generate response using OpenAI GPT-4V"""
        try:
            if not self.api_key:
                return "Error: No OpenAI API key provided"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Prepare message content
            content = [{"type": "text", "text": prompt}]
            
            # Add image if provided
            if image is not None:
                # Convert image to base64
                if isinstance(image, str):
                    # Assume it's a file path
                    with open(image, "rb") as img_file:
                        image_data = base64.b64encode(img_file.read()).decode()
                elif isinstance(image, bytes):
                    image_data = base64.b64encode(image).decode()
                elif isinstance(image, Image.Image):
                    buffer = io.BytesIO()
                    image.save(buffer, format="JPEG")
                    image_data = base64.b64encode(buffer.getvalue()).decode()
                else:
                    logger.error(f"Unsupported image type: {type(image)}")
                    return "Error: Unsupported image type"
                
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}",
                        "detail": "low"  # For faster processing
                    }
                })
            
            messages = [{"role": "user", "content": content}]
            
            data = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            response = requests.post(self.base_url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"Error: {e}"
    
    def is_loaded(self) -> bool:
        """Check if API is accessible"""
        return bool(self.api_key)
    
    def unload(self):
        """No-op for API models"""
        pass
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """API models don't use local memory"""
        return {"type": "api", "memory_mb": 0}


class AnthropicClaudeVisionModel:
    """Anthropic Claude 3 with vision capabilities"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "claude-3-sonnet-20240229"):
        """
        Initialize Anthropic API client.
        
        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model_name: Model to use (claude-3-sonnet, claude-3-opus, etc.)
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model_name = model_name
        self.base_url = "https://api.anthropic.com/v1/messages"
        
        if not self.api_key:
            logger.warning("No Anthropic API key provided. Set ANTHROPIC_API_KEY environment variable.")
    
    def generate_response(
        self, 
        prompt: str, 
        image: Optional[Union[Image.Image, str, bytes]] = None,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """Generate response using Claude 3 Vision"""
        try:
            if not self.api_key:
                return "Error: No Anthropic API key provided"
            
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
            
            # Prepare message content
            content = []
            
            # Add image if provided
            if image is not None:
                # Convert image to base64
                if isinstance(image, str):
                    with open(image, "rb") as img_file:
                        image_data = base64.b64encode(img_file.read()).decode()
                elif isinstance(image, bytes):
                    image_data = base64.b64encode(image).decode()
                elif isinstance(image, Image.Image):
                    buffer = io.BytesIO()
                    image.save(buffer, format="JPEG")
                    image_data = base64.b64encode(buffer.getvalue()).decode()
                else:
                    logger.error(f"Unsupported image type: {type(image)}")
                    return "Error: Unsupported image type"
                
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data
                    }
                })
            
            content.append({"type": "text", "text": prompt})
            
            data = {
                "model": self.model_name,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": content}]
            }
            
            response = requests.post(self.base_url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result["content"][0]["text"]
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return f"Error: {e}"
    
    def is_loaded(self) -> bool:
        """Check if API is accessible"""
        return bool(self.api_key)
    
    def unload(self):
        """No-op for API models"""
        pass
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """API models don't use local memory"""
        return {"type": "api", "memory_mb": 0}