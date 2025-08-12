"""
Test script for the RAG models with smaller/simpler models first
"""

import sys
sys.path.append('/Users/vivitsashankar/Desktop/workspace/drive-qa')

from rag.models.llama_model import LlamaModel
from loguru import logger

def test_llama_model():
    """Test Llama model with a smaller model first"""
    try:
        logger.info("Testing Llama model...")
        
        # Use a smaller model for testing
        model_name = "microsoft/DialoGPT-small"  # Much smaller than Llama 3
        
        llama = LlamaModel(
            model_name=model_name,
            device="cpu",  # Use CPU for testing
            quantization=None
        )
        
        response = llama.generate_response(
            "What is the weather like today?",
            max_tokens=50,
            temperature=0.7
        )
        
        print(f"Response: {response}")
        
        llama.unload()
        logger.info("Llama model test completed successfully")
        
    except Exception as e:
        logger.error(f"Llama model test failed: {e}")
        return False
    
    return True

def main():
    """Run model tests"""
    logger.info("Starting model tests...")
    
    # Test Llama first
    llama_success = test_llama_model()
    
    if llama_success:
        print("✅ All model tests passed!")
    else:
        print("❌ Some model tests failed")

if __name__ == "__main__":
    main()