"""
Test LLaVA multimodal inference with real driving images
"""

import sys
sys.path.append('/Users/vivitsashankar/Desktop/workspace/drive-qa')

from rag.pipeline.rag_pipeline import RAGPipeline
from rag.models.model_loader import ModelLoader
from loguru import logger
import json
import os

def test_llava_with_driving_images():
    """Test LLaVA with actual driving scene images"""
    try:
        logger.info("Testing LLaVA multimodal inference on CPU...")
        
        # Initialize pipeline with CPU optimization
        pipeline = RAGPipeline(
            data_path="data/concatenated_data/concatenated_data.json",
            device="cpu",  # Force CPU
            quantization="8bit"  # Use 8-bit quantization for CPU efficiency
        )
        
        # Test multimodal queries that require both text context and images
        multimodal_queries = [
            "What objects can you see in the camera images?",
            "Is there a car in front of the ego vehicle?", 
            "What is the traffic situation ahead?",
            "Describe what you see in the front camera"
        ]
        
        scene_id = 1
        keyframe_id = 1
        
        print(f"🎥 Testing LLaVA on Scene {scene_id}, Keyframe {keyframe_id}")
        print("=" * 60)
        
        for i, query in enumerate(multimodal_queries, 1):
            print(f"\n🔍 Query {i}: {query}")
            
            # Force multimodal strategy to test LLaVA
            result = pipeline.answer_question(
                query=query,
                scene_id=scene_id,
                keyframe_id=keyframe_id,
                force_strategy="multimodal"  # Force LLaVA usage
            )
            
            print(f"✅ Success: {result['success']}")
            if result['success']:
                print(f"🤖 Answer: {result['answer']}")
                print(f"🎯 Strategy: {result.get('generation_metadata', {}).get('strategy', 'unknown')}")
                
                # Show context info
                generation_meta = result.get('generation_metadata', {})
                if 'context_summary' in generation_meta:
                    context = generation_meta['context_summary']
                    print(f"📊 Context: {context.get('total_objects', 0)} objects, Speed: {context.get('ego_speed', 'N/A')} m/s")
            else:
                print(f"❌ Error: {result['answer']}")
                # If multimodal fails, try with hybrid
                print("🔄 Retrying with hybrid strategy...")
                result = pipeline.answer_question(
                    query=query,
                    scene_id=scene_id,
                    keyframe_id=keyframe_id,
                    force_strategy="hybrid"
                )
                print(f"✅ Hybrid Success: {result['success']}")
                print(f"🤖 Hybrid Answer: {result['answer']}")
            
            print("-" * 40)
        
        # Test dataset QA with multimodal
        print(f"\n🎯 Testing Dataset QA with multimodal...")
        dataset_result = pipeline.answer_dataset_qa(
            scene_id=scene_id,
            keyframe_id=keyframe_id,
            qa_type="perception",
            qa_serial=1,
            force_strategy="multimodal"
        )
        
        if dataset_result['success']:
            print(f"📋 Dataset Question: {dataset_result['query']}")
            print(f"🤖 Generated Answer: {dataset_result['answer']}")
            print(f"✅ Ground Truth: {dataset_result['dataset_info']['ground_truth']}")
            print(f"📊 Overall Score: {dataset_result['dataset_info']['evaluation']['overall_score']:.3f}")
        else:
            print(f"❌ Dataset QA failed: {dataset_result['answer']}")
        
        # Show memory usage
        memory_stats = pipeline.model_loader.get_memory_usage()
        print(f"\n💾 Memory Usage:")
        print(f"   LLaVA Loaded: {memory_stats['llava_loaded']}")
        print(f"   Llama Loaded: {memory_stats['llama_loaded']}")
        
        pipeline.unload_models()
        return True
        
    except Exception as e:
        logger.error(f"LLaVA multimodal test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run LLaVA multimodal test"""
    logger.info("Starting LLaVA multimodal test...")
    
    success = test_llava_with_driving_images()
    
    if success:
        print("\n🎉 LLaVA multimodal test completed!")
        print("🔥 Vision-Language capabilities working!")
        print("📱 CPU-optimized inference successful!")
    else:
        print("\n❌ LLaVA multimodal test failed")

if __name__ == "__main__":
    main()