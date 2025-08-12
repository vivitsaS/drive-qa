"""
Demonstration of Dual-Mode RAG Pipeline

Shows both custom questions and dataset QA evaluation modes.
"""

from rag.pipeline.rag_pipeline import RAGPipeline
from loguru import logger


def demonstrate_custom_mode(pipeline: RAGPipeline):
    """Demonstrate custom question mode"""
    print("🤖 CUSTOM QUESTION MODE")
    print("=" * 50)
    
    custom_questions = [
        "Is there a car in front of the ego vehicle?",
        "What is the ego vehicle's current speed?", 
        "Should the ego vehicle brake or continue?",
        "What objects are detected by the sensors?"
    ]
    
    for i, question in enumerate(custom_questions, 1):
        print(f"\n{i}. Question: {question}")
        
        result = pipeline.answer_question(
            query=question,
            scene_id=1,
            keyframe_id=3,
            force_strategy="hybrid"
        )
        
        if result["success"]:
            print(f"   Answer: {result['answer']}")
            print(f"   Strategy: {result['generation_metadata']['strategy']}")
            print(f"   Confidence: {result['generation_metadata']['confidence']}")
        else:
            print(f"   Error: {result.get('error', 'Unknown error')}")


def demonstrate_dataset_mode(pipeline: RAGPipeline):
    """Demonstrate dataset QA evaluation mode"""
    print("\n📊 DATASET QA EVALUATION MODE")
    print("=" * 50)
    
    # Test specific dataset QA pairs
    test_cases = [
        {"scene_id": 1, "keyframe_id": 1, "qa_type": "perception", "qa_serial": 1},
        {"scene_id": 1, "keyframe_id": 1, "qa_type": "perception", "qa_serial": 2},
        {"scene_id": 1, "keyframe_id": 2, "qa_type": "planning", "qa_serial": 1},
        {"scene_id": 1, "keyframe_id": 3, "qa_type": "prediction", "qa_serial": 1}
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing {test_case['qa_type']} question #{test_case['qa_serial']}")
        print(f"   Scene: {test_case['scene_id']}, Keyframe: {test_case['keyframe_id']}")
        
        result = pipeline.answer_dataset_qa(**test_case, force_strategy="hybrid")
        
        if result["success"]:
            dataset_info = result["dataset_info"]
            evaluation = dataset_info["evaluation"]
            
            print(f"   Question: {result['query']}")
            print(f"   Generated: {result['answer']}")
            print(f"   Ground Truth: {dataset_info['ground_truth']}")
            print(f"   📈 Metrics:")
            print(f"      - Overall Score: {evaluation['overall_score']:.3f}")
            print(f"      - Word F1: {evaluation['word_f1']:.3f}")
            print(f"      - Exact Match: {evaluation['exact_match']}")
            print(f"      - Semantic Similarity: {evaluation['semantic_similarity']:.3f}")
            
            # Show QA-type specific metrics if available
            qa_specific = {k: v for k, v in evaluation.items() 
                         if k.endswith('_accuracy') and k != 'accuracy_score'}
            if qa_specific:
                print(f"      - QA-Specific: {qa_specific}")
        else:
            print(f"   ❌ Error: {result.get('error', 'Unknown error')}")


def demonstrate_batch_evaluation(pipeline: RAGPipeline):
    """Demonstrate batch evaluation capabilities"""
    print("\n🚀 BATCH EVALUATION MODE")
    print("=" * 50)
    
    # Create a comprehensive test set
    test_specifications = []
    
    # Add perception questions
    for scene_id in [1, 1]:
        for keyframe_id in [1, 2, 3]:
            for qa_serial in [1, 2]:
                test_specifications.append({
                    "scene_id": scene_id,
                    "keyframe_id": keyframe_id,
                    "qa_type": "perception",
                    "qa_serial": qa_serial
                })
    
    # Add other QA types  
    for qa_type in ["planning", "prediction", "behavior"]:
        test_specifications.append({
            "scene_id": 1,
            "keyframe_id": 1,
            "qa_type": qa_type,
            "qa_serial": 1
        })
    
    print(f"Evaluating {len(test_specifications)} QA pairs...")
    
    batch_results = pipeline.evaluate_dataset_qa_batch(
        qa_specifications=test_specifications,
        force_strategy="hybrid"
    )
    
    metrics = batch_results["evaluation_metrics"]
    
    print(f"\n📊 BATCH RESULTS:")
    print(f"   Total QA pairs: {metrics['total_pairs']}")
    print(f"   Success rate: {metrics['success_rate']:.1%}")
    print(f"   Average overall score: {metrics['avg_overall_score']:.3f}")
    print(f"   Average semantic similarity: {metrics['avg_semantic_similarity']:.3f}")
    
    print(f"\n📈 BY QA TYPE:")
    for qa_type, type_data in metrics['by_qa_type'].items():
        count = type_data['count']
        score = type_data.get('avg_overall_score', 0)
        similarity = type_data.get('avg_semantic_similarity', 0)
        print(f"   {qa_type.capitalize()}: {count} pairs, "
              f"score: {score:.3f}, similarity: {similarity:.3f}")


def demonstrate_comparison(pipeline: RAGPipeline):
    """Demonstrate comparison between modes"""
    print("\n🔄 MODE COMPARISON")
    print("=" * 50)
    
    # Use the same scene/keyframe for fair comparison
    scene_id, keyframe_id = 1, 1
    
    # Get a dataset question
    dataset_result = pipeline.answer_dataset_qa(
        scene_id=scene_id,
        keyframe_id=keyframe_id,
        qa_type="perception",
        qa_serial=1,
        force_strategy="hybrid"
    )
    
    if dataset_result["success"]:
        original_question = dataset_result["query"]
        ground_truth = dataset_result["dataset_info"]["ground_truth"]
        dataset_answer = dataset_result["answer"]
        
        print(f"Original Dataset Question: {original_question}")
        print(f"Ground Truth Answer: {ground_truth}")
        print(f"RAG Dataset Mode Answer: {dataset_answer}")
        
        # Now ask the same question in custom mode
        custom_result = pipeline.answer_question(
            query=original_question,
            scene_id=scene_id,
            keyframe_id=keyframe_id,
            force_strategy="hybrid"
        )
        
        if custom_result["success"]:
            custom_answer = custom_result["answer"]
            print(f"RAG Custom Mode Answer: {custom_answer}")
            
            # Compare the two RAG answers
            print(f"\n🔍 COMPARISON:")
            print(f"   Same answers: {dataset_answer.strip() == custom_answer.strip()}")
            print(f"   Dataset mode strategy: {dataset_result['generation_metadata']['strategy']}")
            print(f"   Custom mode strategy: {custom_result['generation_metadata']['strategy']}")
            
            # Evaluate custom answer against ground truth
            custom_evaluation = pipeline.evaluator.evaluate_answer(
                custom_answer, ground_truth, "perception"
            )
            dataset_evaluation = dataset_result["dataset_info"]["evaluation"]
            
            print(f"   Dataset mode score: {dataset_evaluation['overall_score']:.3f}")
            print(f"   Custom mode score: {custom_evaluation['overall_score']:.3f}")


def main():
    """Main demonstration function"""
    print("🚗 RAG PIPELINE DUAL-MODE DEMONSTRATION")
    print("=" * 60)
    
    # Initialize pipeline (LLaVA + Llama 3)
    logger.info("Initializing RAG Pipeline...")
    pipeline = RAGPipeline(
        data_path="data/concatenated_data/concatenated_data.json",
        device="auto",
        quantization=None
    )
    
    try:
        # Demonstrate custom mode
        demonstrate_custom_mode(pipeline)
        
        # Demonstrate dataset mode
        demonstrate_dataset_mode(pipeline)
        
        # Demonstrate batch evaluation
        demonstrate_batch_evaluation(pipeline)
        
        # Demonstrate comparison
        demonstrate_comparison(pipeline)
        
        # Show final statistics
        stats = pipeline.get_pipeline_stats()
        print(f"\n📈 FINAL PIPELINE STATISTICS:")
        print(f"   Total queries processed: {stats['queries_processed']}")
        print(f"   Success rate: {stats['success_rate']:.1%}")
        print(f"   Multimodal queries: {stats['multimodal_queries']}")
        print(f"   Text-only queries: {stats['text_only_queries']}")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"❌ Error: {e}")
    
    finally:
        # Cleanup
        pipeline.unload_models()
        logger.info("Demonstration completed")


if __name__ == "__main__":
    main()