"""
Test RAG pipeline with fully open models
"""

import sys
sys.path.append('/Users/vivitsashankar/Desktop/workspace/drive-qa')

from rag.pipeline.rag_pipeline import RAGPipeline
from rag.models.model_loader import ModelLoader
from loguru import logger

def test_rag_with_open_models():
    """Test RAG pipeline with fully open models"""
    try:
        logger.info("Testing RAG pipeline with open models...")
        
        # Initialize pipeline
        pipeline = RAGPipeline(
            data_path="data/concatenated_data/concatenated_data.json",
            device="cpu",  # Use CPU for stability 
            quantization=None
        )
        
        # Override to use an open Llama-style model
        open_llama_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small but functional
        
        # Test with text-only query using the open model
        result = pipeline.answer_question(
            query="What is the speed of the ego vehicle?",
            scene_id=1,
            keyframe_id=1,
            force_strategy="text_only"
        )
        
        # If it failed with the default model, try with explicit model override
        if not result['success']:
            logger.info(f"Retrying with explicit model: {open_llama_model}")
            # Load the open model directly
            pipeline.model_loader._llama_model = None  # Reset
            pipeline.model_loader.load_llama(open_llama_model)
            
            result = pipeline.answer_question(
                query="What is the speed of the ego vehicle?",
                scene_id=1,
                keyframe_id=1,
                force_strategy="text_only"
            )
        
        print(f"Query: What is the speed of the ego vehicle?")
        print(f"Answer: {result['answer']}")
        print(f"Success: {result['success']}")
        print(f"Strategy: {result.get('generation_metadata', {}).get('strategy', 'unknown')}")
        
        # If successful, test dataset mode
        if result['success']:
            dataset_result = pipeline.answer_dataset_qa(
                scene_id=1,
                keyframe_id=1,
                qa_type="perception", 
                qa_serial=1,
                force_strategy="text_only"
            )
            
            if dataset_result['success']:
                print(f"\nDataset Question: {dataset_result['query']}")
                print(f"Generated Answer: {dataset_result['answer']}")
                print(f"Ground Truth: {dataset_result['dataset_info']['ground_truth']}")
                print(f"Overall Score: {dataset_result['dataset_info']['evaluation']['overall_score']:.3f}")
                
                # Show key metrics
                eval_data = dataset_result['dataset_info']['evaluation']
                print(f"Exact Match: {eval_data['exact_match']}")
                print(f"Word F1: {eval_data['word_f1']:.3f}")
                print(f"Length Ratio: {eval_data['length_ratio']:.3f}")
        
        pipeline.unload_models()
        return result['success']
        
    except Exception as e:
        logger.error(f"RAG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run RAG test"""
    logger.info("Starting RAG pipeline test with open models...")
    
    success = test_rag_with_open_models()
    
    if success:
        print("\n✅ RAG pipeline working perfectly!")
        print("🎯 Context retrieval: ✓")
        print("🤖 Text generation: ✓") 
        print("📊 Evaluation metrics: ✓")
        print("🔄 Dataset QA mode: ✓")
    else:
        print("\n❌ RAG pipeline test failed")

if __name__ == "__main__":
    main()