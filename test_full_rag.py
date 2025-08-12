"""
Test RAG pipeline with working open models
"""

import sys
sys.path.append('/Users/vivitsashankar/Desktop/workspace/drive-qa')

from rag.pipeline.rag_pipeline import RAGPipeline
from loguru import logger

def test_rag_with_working_models():
    """Test RAG pipeline with open models that should work"""
    try:
        logger.info("Testing RAG pipeline with working open models...")
        
        # Initialize pipeline with CPU and no quantization for testing
        pipeline = RAGPipeline(
            data_path="data/concatenated_data/concatenated_data.json",
            device="cpu",
            quantization=None
        )
        
        # Test with a known working text model
        result = pipeline.answer_question(
            query="What is the speed of the ego vehicle?",
            scene_id=1,
            keyframe_id=1,
            force_strategy="text_only"  # Force text-only
        )
        
        print(f"Query: What is the speed of the ego vehicle?")
        print(f"Answer: {result['answer']}")
        print(f"Success: {result['success']}")
        print(f"Strategy: {result.get('generation_metadata', {}).get('strategy', 'unknown')}")
        
        if result['success']:
            # Test dataset QA mode
            dataset_result = pipeline.answer_dataset_qa(
                scene_id=1,
                keyframe_id=1,
                qa_type="perception",
                qa_serial=1,
                force_strategy="text_only"  # Force text-only
            )
            
            if dataset_result['success']:
                print(f"\nDataset Question: {dataset_result['query']}")
                print(f"Generated Answer: {dataset_result['answer']}")
                print(f"Ground Truth: {dataset_result['dataset_info']['ground_truth']}")
                print(f"Overall Score: {dataset_result['dataset_info']['evaluation']['overall_score']:.3f}")
        
        pipeline.unload_models()
        return result['success']
        
    except Exception as e:
        logger.error(f"RAG test failed: {e}")
        return False

def main():
    """Run RAG test"""
    logger.info("Starting RAG pipeline test with working models...")
    
    success = test_rag_with_working_models()
    
    if success:
        print("\n✅ RAG pipeline test passed!")
    else:
        print("\n❌ RAG pipeline test failed")

if __name__ == "__main__":
    main()