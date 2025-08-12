"""
Llama 3 Model Wrapper

Handles Llama 3 model loading and inference for text reasoning.
"""

from typing import Optional, List
import torch
from loguru import logger


class LlamaModel:
    """Llama 3 model wrapper for text reasoning"""
    
    def __init__(self, model_name: str, device: str = "auto", quantization: Optional[str] = None):
        """
        Initialize Llama 3 model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on
            quantization: Quantization configuration
        """
        self.model_name = model_name
        self.device = device
        self.quantization = quantization
        self.model = None
        self.tokenizer = None
        
        # TODO: Load model components
        self._load_model()
        
    def _load_model(self):
        """Load Llama 3 model components"""
        try:
            logger.info(f"Loading Llama 3 model: {self.model_name}")
            
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Configure model loading based on quantization
            model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": self.device if self.device != "auto" else "auto",
            }
            
            if self.quantization == "4bit":
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            elif self.quantization == "8bit":
                model_kwargs["load_in_8bit"] = True
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Llama 3 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Llama 3 model: {e}")
            raise
            
    def generate_response(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """
        Generate response using Llama 3 model.
        
        Args:
            prompt: User prompt/question
            system_prompt: System prompt for instruction following
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        try:
            logger.info("Generating Llama 3 response")
            
            if not self.is_loaded():
                return "Error: Llama 3 model not loaded"
            
            # Prepare messages for chat template
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Apply chat template
            try:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except Exception:
                # Fallback if chat template fails
                if system_prompt:
                    formatted_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
                else:
                    formatted_prompt = f"User: {prompt}\n\nAssistant:"
            
            # Tokenize
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            
            # Move to device if needed
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                generate_kwargs = {
                    "max_new_tokens": max_tokens,
                    "do_sample": temperature > 0,
                    "pad_token_id": self.tokenizer.eos_token_id,
                }
                
                if temperature > 0:
                    generate_kwargs.update({
                        "temperature": temperature,
                        "top_p": 0.9,
                        "top_k": 50,
                    })
                
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    **generate_kwargs
                )
            
            # Decode response (only the new tokens)
            generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            logger.info(f"Generated Llama 3 response: {response[:100]}...")
            return response.strip()
            
        except Exception as e:
            logger.error(f"Llama 3 generation failed: {e}")
            return f"Error: {e}"
            
    def analyze_structured_data(self, data: dict, question: str) -> str:
        """
        Analyze structured data to answer questions.
        
        Args:
            data: Structured data (vehicle state, annotations, etc.)
            question: Question about the data
            
        Returns:
            Analysis result
        """
        # Format structured data for analysis
        data_str = self._format_structured_data(data)
        
        system_prompt = """You are an expert at analyzing autonomous driving data. 
        Given structured vehicle and sensor data, answer questions accurately and concisely."""
        
        prompt = f"""Data:
{data_str}

Question: {question}

Answer:"""
        
        return self.generate_response(prompt, system_prompt)
        
    def _format_structured_data(self, data: dict) -> str:
        """Format structured data for model consumption"""
        # TODO: Implement smart data formatting
        return str(data)
        
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
        
    def unload(self):
        """Unload model from memory"""
        if self.model:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            
        self.model = None
        self.tokenizer = None
        
        logger.info("Llama 3 model unloaded")
        
    def get_memory_usage(self) -> dict:
        """Get model memory usage"""
        # TODO: Implement memory tracking
        return {
            "model_loaded": self.is_loaded(),
            "estimated_memory_mb": 10000 if self.is_loaded() else 0  # 8B model ~10GB
        }