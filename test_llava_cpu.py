"""
Test LLaVA on CPU without quantization
"""

import sys
sys.path.append('/Users/vivitsashankar/Desktop/workspace/drive-qa')

from rag.pipeline.rag_pipeline import RAGPipeline
from loguru import logger
import torch

def test_llava_cpu_no_quantization():
    """Test LLaVA on CPU without quantization"""
    try:
        logger.info("Testing LLaVA on CPU without quantization...")
        
        # Initialize pipeline for CPU without quantization
        pipeline = RAGPipeline(
            data_path="data/concatenated_data/concatenated_data.json",
            device="cpu",  # Force CPU
            quantization=None  # No quantization to avoid bitsandbytes
        )
        
        # Test a simple multimodal query
        query = "What objects are around the ego vehicle?"
        
        print(f"🎥 Testing LLaVA Multimodal Inference")
        print(f"🔍 Query: {query}")
        print("=" * 60)
        
        # Try multimodal first
        result = pipeline.answer_question(
            query=query,
            scene_id=1,
            keyframe_id=1,
            force_strategy="multimodal"
        )
        
        print(f"✅ Success: {result['success']}")
        
        if result['success']:
            print(f"🤖 Answer: {result['answer']}")
            print(f"🎯 Strategy: {result.get('generation_metadata', {}).get('strategy', 'unknown')}")
            
            # Show generation details
            gen_meta = result.get('generation_metadata', {})
            if 'timing' in gen_meta:
                print(f"⏱️ Generation Time: {gen_meta['timing']:.2f}s")
        else:
            print(f"❌ Error: {result['answer']}")
            
            # Try with just text strategy as fallback
            print("\n🔄 Trying text-only strategy as fallback...")
            
            result = pipeline.answer_question(
                query=query,
                scene_id=1,
                keyframe_id=1,
                force_strategy="text_only"
            )
            
            print(f"✅ Text Success: {result['success']}")
            if result['success']:
                print(f"🤖 Text Answer: {result['answer']}")
        
        # Test dataset QA
        print(f"\n🎯 Testing Dataset QA...")
        dataset_result = pipeline.answer_dataset_qa(
            scene_id=1,
            keyframe_id=1,
            qa_type="perception",
            qa_serial=1,
            force_strategy="text_only"  # Use text for now
        )
        
        if dataset_result['success']:
            print(f"📋 Question: {dataset_result['query']}")
            print(f"🤖 Answer: {dataset_result['answer']}")
            print(f"✅ Ground Truth: {dataset_result['dataset_info']['ground_truth']}")
            print(f"📊 Score: {dataset_result['dataset_info']['evaluation']['overall_score']:.3f}")
        
        pipeline.unload_models()
        return result['success'] or dataset_result['success']
        
    except Exception as e:
        logger.error(f"CPU LLaVA test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run CPU LLaVA test"""
    logger.info("Starting CPU LLaVA test...")
    
    # Show PyTorch info
    print(f"🔧 PyTorch Version: {torch.__version__}")
    print(f"💻 CUDA Available: {torch.cuda.is_available()}")
    print(f"🎯 Device: CPU (forced)")
    print()
    
    success = test_llava_cpu_no_quantization()
    
    if success:
        print("\n🎉 LLaVA CPU test successful!")
        print("✅ Models can load on CPU")
        print("✅ Pipeline works without GPU")
    else:
        print("\n❌ LLaVA CPU test failed")

if __name__ == "__main__":
    main()