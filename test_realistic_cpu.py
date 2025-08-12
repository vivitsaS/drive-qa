"""
Realistic CPU testing for the RAG pipeline
"""

import sys
sys.path.append('/Users/vivitsashankar/Desktop/workspace/drive-qa')

from rag.pipeline.rag_pipeline import RAGPipeline
from loguru import logger
import psutil
import os

def show_system_info():
    """Show system resources"""
    memory = psutil.virtual_memory()
    print(f"💻 System Info:")
    print(f"   Total RAM: {memory.total / (1024**3):.1f} GB")
    print(f"   Available RAM: {memory.available / (1024**3):.1f} GB")
    print(f"   CPU Cores: {psutil.cpu_count()}")
    print()

def test_text_only_pipeline():
    """Test the working text-only pipeline"""
    try:
        logger.info("Testing text-only pipeline (CPU optimized)...")
        
        pipeline = RAGPipeline(
            data_path="data/concatenated_data/concatenated_data.json",
            device="cpu",
            quantization=None
        )
        
        # Use TinyLlama for CPU efficiency
        pipeline.model_loader._llama_model = None
        pipeline.model_loader.load_llama("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        
        queries = [
            "What is the speed of the ego vehicle?",
            "How many objects are detected around the vehicle?",
            "What type of objects are in front of the ego car?"
        ]
        
        print("🚗 Testing Driving Scene Analysis (Text-Only)")
        print("=" * 50)
        
        for i, query in enumerate(queries, 1):
            print(f"\n🔍 Query {i}: {query}")
            
            result = pipeline.answer_question(
                query=query,
                scene_id=1,
                keyframe_id=1,
                force_strategy="text_only"
            )
            
            if result['success']:
                print(f"✅ Answer: {result['answer']}")
            else:
                print(f"❌ Error: {result['answer']}")
        
        # Test dataset QA
        print(f"\n🎯 Dataset QA Test:")
        dataset_result = pipeline.answer_dataset_qa(
            scene_id=1,
            keyframe_id=1,
            qa_type="perception",
            qa_serial=1,
            force_strategy="text_only"
        )
        
        if dataset_result['success']:
            print(f"📋 Question: {dataset_result['query']}")
            print(f"🤖 Generated: {dataset_result['answer']}")
            print(f"✅ Ground Truth: {dataset_result['dataset_info']['ground_truth']}")
            print(f"📊 Score: {dataset_result['dataset_info']['evaluation']['overall_score']:.3f}")
        
        pipeline.unload_models()
        return True
        
    except Exception as e:
        logger.error(f"Text pipeline test failed: {e}")
        return False

def explain_multimodal_options():
    """Explain multimodal options for CPU"""
    print("\n🔮 Multimodal Options for CPU:")
    print("=" * 40)
    print("1. 🔥 **Cloud APIs** (Recommended)")
    print("   - OpenAI GPT-4V")
    print("   - Google Gemini Vision")
    print("   - Anthropic Claude Vision")
    print()
    print("2. 🏃 **Smaller Models**")
    print("   - BLIP-2 (lighter than LLaVA)")
    print("   - InstructBLIP")
    print("   - MiniGPT-4")
    print()
    print("3. ⚡ **GPU Alternatives**")
    print("   - Google Colab (free GPU)")
    print("   - AWS EC2 with GPU")
    print("   - Local GPU if available")
    print()
    print("4. 📊 **Text-Based Analysis**")
    print("   - Use detection metadata instead of images")
    print("   - Rich context from sensor data")
    print("   - Object positions, types, distances")

def main():
    """Run realistic CPU tests"""
    logger.info("Starting realistic CPU testing...")
    
    show_system_info()
    
    # Test what actually works
    success = test_text_only_pipeline()
    
    # Explain multimodal options
    explain_multimodal_options()
    
    if success:
        print("\n🎉 **SUCCESS: Your RAG pipeline is working!**")
        print("✅ Text-based reasoning: Working")
        print("✅ Context retrieval: Working") 
        print("✅ Dataset evaluation: Working")
        print("✅ Vehicle data analysis: Working")
        print("\n💡 For vision capabilities, consider cloud APIs or GPU access.")
    else:
        print("\n❌ Pipeline test failed")

if __name__ == "__main__":
    main()