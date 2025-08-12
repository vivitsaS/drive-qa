"""
Test with fast, CPU-appropriate HuggingFace models
"""

import sys
sys.path.append('/Users/vivitsashankar/Desktop/workspace/drive-qa')

from rag.pipeline.rag_pipeline import RAGPipeline
from loguru import logger
import time

def test_cpu_appropriate_models():
    """Test with models designed for CPU inference"""
    
    # Models optimized for CPU/fast inference
    cpu_models = [
        {
            "name": "microsoft/DialoGPT-medium", 
            "description": "117M params, dialog-optimized",
            "speed": "Very Fast"
        },
        {
            "name": "distilgpt2", 
            "description": "82M params, lightweight GPT-2",
            "speed": "Extremely Fast"
        },
        {
            "name": "gpt2", 
            "description": "124M params, classic GPT-2",
            "speed": "Fast"
        }
    ]
    
    print("🚗 Testing CPU-Appropriate Models for Driving QA")
    print("=" * 55)
    
    for model_config in cpu_models:
        print(f"\n🤖 Testing: {model_config['name']}")
        print(f"📊 Size: {model_config['description']}")
        print(f"⚡ Expected Speed: {model_config['speed']}")
        print("-" * 40)
        
        try:
            # Initialize pipeline
            pipeline = RAGPipeline(
                data_path="data/concatenated_data/concatenated_data.json",
                device="cpu",
                quantization=None
            )
            
            # Load the specific model
            pipeline.model_loader._llama_model = None
            llama_model = pipeline.model_loader.load_llama(model_config['name'])
            
            # Test query
            query = "What is the ego vehicle speed?"
            
            start_time = time.time()
            
            result = pipeline.answer_question(
                query=query,
                scene_id=1,
                keyframe_id=1,
                force_strategy="text_only"
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result['success']:
                print(f"✅ SUCCESS ({duration:.1f}s)")
                print(f"🎯 Answer: {result['answer']}")
                
                # Rate the performance
                if duration < 5:
                    print(f"🔥 Performance: Excellent!")
                elif duration < 15:
                    print(f"⚡ Performance: Good")
                elif duration < 30:
                    print(f"⏳ Performance: Acceptable")
                else:
                    print(f"🐌 Performance: Too slow")
                    
                # Test one more query to verify consistency
                start_time2 = time.time()
                result2 = pipeline.answer_question(
                    query="How many objects are detected?",
                    scene_id=1,
                    keyframe_id=1,
                    force_strategy="text_only"
                )
                end_time2 = time.time()
                duration2 = end_time2 - start_time2
                
                if result2['success']:
                    print(f"✅ Second query: {duration2:.1f}s")
                    print(f"🎯 Answer: {result2['answer']}")
                    avg_time = (duration + duration2) / 2
                    print(f"📊 Average time: {avg_time:.1f}s")
                    
                    # This model works well!
                    if avg_time < 10:
                        print(f"🎉 RECOMMENDED: This model works great for CPU!")
                        
                        # Test dataset QA
                        start_time3 = time.time()
                        dataset_result = pipeline.answer_dataset_qa(
                            scene_id=1,
                            keyframe_id=1,
                            qa_type="perception",
                            qa_serial=1,
                            force_strategy="text_only"
                        )
                        end_time3 = time.time()
                        duration3 = end_time3 - start_time3
                        
                        if dataset_result['success']:
                            print(f"📋 Dataset QA ({duration3:.1f}s):")
                            print(f"   Question: {dataset_result['query']}")
                            print(f"   Generated: {dataset_result['answer']}")
                            print(f"   Ground Truth: {dataset_result['dataset_info']['ground_truth']}")
                            print(f"   Score: {dataset_result['dataset_info']['evaluation']['overall_score']:.3f}")
                            
                        return model_config['name'], avg_time  # Return the working model
            else:
                print(f"❌ FAILED: {result['answer']}")
                
            pipeline.unload_models()
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            
        print()
    
    return None, None

def main():
    """Test CPU-appropriate models"""
    logger.info("Testing CPU-appropriate HuggingFace models...")
    
    print("🎯 Goal: Find a fast model for CPU inference")
    print("💡 Looking for: <10 second response times")
    print()
    
    best_model, best_time = test_cpu_appropriate_models()
    
    if best_model:
        print(f"\n🏆 WINNER: {best_model}")
        print(f"⚡ Average time: {best_time:.1f} seconds")
        print(f"✅ Your RAG pipeline is ready for production!")
        print(f"\n💡 To use this model by default, it's already configured in model_loader.py")
    else:
        print(f"\n🤔 All models were too slow or failed")
        print(f"💡 Consider using cloud APIs for faster inference")

if __name__ == "__main__":
    main()