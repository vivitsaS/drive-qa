"""
Example Usage of RAG Pipeline

Demonstrates how to use the complete RAG system for answering
autonomous driving questions.
"""

from rag.pipeline.rag_pipeline import RAGPipeline
from loguru import logger


def main():
    """Demonstrate RAG pipeline usage"""
    
    # Initialize pipeline (now using LLaVA + Llama 3)
    logger.info("Initializing RAG Pipeline...")
    pipeline = RAGPipeline(
        data_path="data/concatenated_data/concatenated_data.json",
        device="auto",  # or "cuda" if GPU available
        quantization=None  # or "4bit" for memory efficiency
    )
    
    print("="*80)
    print("DEMO 1: Custom Questions Mode")
    print("="*80)
    
    # Example custom queries
    custom_queries = [
        "Is the car in front of me braking?",
        "What objects are visible in the front camera?",
        "What is the current speed of the ego vehicle?",
        "Are there any pedestrians near the vehicle?"
    ]
    
    logger.info("Processing custom queries...")
    
    # Process single custom query
    query = custom_queries[0]
    logger.info(f"Processing custom query: {query}")
    
    result = pipeline.answer_question(
        query=query,
        scene_id=1,  # Optional: specify scene
        keyframe_id=1,  # Optional: specify keyframe
        force_strategy="hybrid"  # Use both visual and text analysis
    )
    
    print(f"\nCustom Query: {query}")
    print(f"Answer: {result['answer']}")
    print(f"Mode: {result['mode']}")
    print(f"Strategy: {result.get('generation_metadata', {}).get('strategy', 'unknown')}")
    print(f"Success: {result['success']}")
    
    print("\n" + "="*80)
    print("DEMO 2: Dataset QA Mode")
    print("="*80)
    
    # Process dataset QA pairs
    logger.info("Processing dataset QA pairs...")
    
    # Single dataset QA pair
    dataset_result = pipeline.answer_dataset_qa(
        scene_id=1,
        keyframe_id=1,
        qa_type="perception",
        qa_serial=1,  # First perception question
        force_strategy="hybrid"
    )
    
    if dataset_result['success']:
        dataset_info = dataset_result['dataset_info']
        print(f"\nDataset Question: {dataset_result['query']}")
        print(f"Generated Answer: {dataset_result['answer']}")
        print(f"Ground Truth: {dataset_info['ground_truth']}")
        print(f"QA Type: {dataset_info['qa_type']}")
        print(f"Overall Score: {dataset_info['evaluation']['overall_score']:.3f}")
        print(f"Word F1: {dataset_info['evaluation']['word_f1']:.3f}")
        print(f"Semantic Similarity: {dataset_info['evaluation']['semantic_similarity']:.3f}")
        print(f"Exact Match: {dataset_info['evaluation']['exact_match']}")
    
    print("\n" + "="*80)
    print("DEMO 3: Batch Dataset Evaluation")
    print("="*80)
    
    # Batch evaluation of dataset QA pairs
    qa_specs = [
        {"scene_id": 1, "keyframe_id": 1, "qa_type": "perception", "qa_serial": 1},
        {"scene_id": 1, "keyframe_id": 1, "qa_type": "perception", "qa_serial": 2},
        {"scene_id": 1, "keyframe_id": 1, "qa_type": "planning", "qa_serial": 1},
        {"scene_id": 1, "keyframe_id": 2, "qa_type": "prediction", "qa_serial": 1}
    ]
    
    batch_eval = pipeline.evaluate_dataset_qa_batch(
        qa_specifications=qa_specs,
        force_strategy="hybrid"
    )
    
    metrics = batch_eval["evaluation_metrics"]
    print(f"\nBatch Evaluation Results:")
    print(f"- Total QA pairs: {metrics['total_pairs']}")
    print(f"- Success rate: {metrics['success_rate']:.2%}")
    print(f"- Average overall score: {metrics['avg_overall_score']:.3f}")
    print(f"- Average semantic similarity: {metrics['avg_semantic_similarity']:.3f}")
    
    print(f"\nBy QA Type:")
    for qa_type, type_data in metrics['by_qa_type'].items():
        print(f"- {qa_type}: {type_data['count']} pairs, "
              f"overall score: {type_data.get('avg_overall_score', 0):.3f}")
    
    # Process batch custom queries
    logger.info("Processing batch custom queries...")
    batch_results = pipeline.batch_answer_questions(
        custom_queries[1:3],
        scene_id=1,
        keyframe_id=2
    )
    
    print(f"\nCustom Batch Results:")
    for i, result in enumerate(batch_results):
        print(f"Query {i+1}: {custom_queries[i+1]}")
        print(f"Answer: {result['answer'][:100]}...")
        print(f"Success: {result['success']}")
        print()
    
    # Show pipeline statistics
    stats = pipeline.get_pipeline_stats()
    print(f"\nPipeline Statistics:")
    print(f"- Queries processed: {stats['queries_processed']}")
    print(f"- Success rate: {stats['success_rate']:.2%}")
    print(f"- Multimodal queries: {stats['multimodal_queries']}")
    print(f"- Text-only queries: {stats['text_only_queries']}")
    
    # Health check
    health = pipeline.health_check()
    print(f"\nPipeline Health: {health['pipeline_status']}")
    
    # Cleanup
    pipeline.unload_models()
    logger.info("Example completed successfully")


if __name__ == "__main__":
    main()