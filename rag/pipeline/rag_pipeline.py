"""
Main RAG Pipeline

Orchestrates the complete RAG workflow: Query Processing -> Context Retrieval -> Generation
"""

from typing import Dict, Any, Optional, List
from loguru import logger

from ..models.model_loader import ModelLoader
from ..retrieval.context_retrieval import ContextualRetriever
from .generation_pipeline import GenerationPipeline
from ..evaluation import RAGEvaluator


class RAGPipeline:
    """Main RAG pipeline orchestrator"""
    
    def __init__(
        self, 
        data_path: str = "data/concatenated_data/concatenated_data.json",
        device: str = "auto",
        quantization: Optional[str] = None
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            data_path: Path to concatenated data file
            device: Device for model loading
            quantization: Model quantization type
        """
        self.data_path = data_path
        
        # Initialize components
        self.model_loader = ModelLoader(device, quantization)
        self.retriever = ContextualRetriever(data_path)
        self.generator = GenerationPipeline(self.model_loader)
        self.evaluator = RAGEvaluator()
        
        # Pipeline statistics
        self.stats = {
            "queries_processed": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "multimodal_queries": 0,
            "text_only_queries": 0
        }
        
        logger.info("RAG Pipeline initialized successfully")
    
    def answer_question(
        self, 
        query: str = None,
        scene_id: Optional[int] = None,
        keyframe_id: Optional[int] = None,
        context_window: int = 3,
        force_strategy: Optional[str] = None,
        # Dataset QA mode parameters
        qa_type: Optional[str] = None,
        qa_serial: Optional[int] = None,
        use_dataset_qa: bool = False
    ) -> Dict[str, Any]:
        """
        Answer a question using the complete RAG pipeline.
        
        Two modes available:
        1. Custom question mode: Provide 'query' parameter
        2. Dataset QA mode: Set use_dataset_qa=True and provide scene_id, keyframe_id, qa_type, qa_serial
        
        Args:
            query: User question (for custom mode)
            scene_id: Target scene ID (required for dataset mode, auto-inferred for custom)
            keyframe_id: Target keyframe ID (required for dataset mode, auto-inferred for custom)
            context_window: Number of keyframes for temporal context
            force_strategy: Force generation strategy ("multimodal", "text_only", "hybrid")
            qa_type: QA type for dataset mode ("perception", "planning", "prediction", "behavior")
            qa_serial: QA pair serial number (1-indexed)
            use_dataset_qa: Whether to use dataset QA pairs (True) or custom query (False)
            
        Returns:
            Complete answer with metadata, provenance, and ground truth (if dataset mode)
        """
        try:
            # Determine mode and validate parameters
            if use_dataset_qa:
                if not all([scene_id, keyframe_id, qa_type, qa_serial]):
                    raise ValueError("Dataset QA mode requires scene_id, keyframe_id, qa_type, and qa_serial")
                
                # Fetch question from dataset
                from ..context_retriever import ContextRetriever
                context_retriever = ContextRetriever(scene_id, keyframe_id)
                qa_pair = context_retriever.get_qa_pair(qa_type, qa_serial)
                
                if qa_pair is None:
                    raise ValueError(f"No QA pair found for {qa_type} #{qa_serial} in scene {scene_id}, keyframe {keyframe_id}")
                
                query = qa_pair["Q"]
                ground_truth = qa_pair["A"]
                logger.info(f"Processing dataset QA: {qa_type} #{qa_serial} - {query}")
            else:
                if query is None:
                    raise ValueError("Custom mode requires 'query' parameter")
                ground_truth = None
                logger.info(f"Processing custom query: {query}")
            
            self.stats["queries_processed"] += 1
            
            # Step 1: Retrieve relevant context
            logger.info("Step 1: Retrieving context...")
            context = self.retriever.retrieve_context(
                query=query,
                scene_id=scene_id,
                keyframe_id=keyframe_id,
                context_window=context_window
            )
            
            if "error" in context:
                self.stats["failed_generations"] += 1
                return {
                    "answer": f"Failed to retrieve context: {context['error']}",
                    "success": False,
                    "error": context["error"],
                    "pipeline_stage": "retrieval"
                }
            
            # Step 2: Generate answer
            logger.info("Step 2: Generating answer...")
            use_multimodal = self._parse_strategy(force_strategy)
            result = self.generator.generate_answer(
                query=query,
                context=context,
                use_multimodal=use_multimodal
            )
            
            if "error" in result:
                self.stats["failed_generations"] += 1
                return {
                    "answer": result["answer"],
                    "success": False,
                    "error": result["error"],
                    "pipeline_stage": "generation"
                }
            
            # Step 3: Package complete response
            self.stats["successful_generations"] += 1
            
            # Update strategy statistics
            if result["strategy"] in ["multimodal", "hybrid"]:
                self.stats["multimodal_queries"] += 1
            else:
                self.stats["text_only_queries"] += 1
            
            complete_response = {
                "answer": result["answer"],
                "success": True,
                "query": query,
                "mode": "dataset_qa" if use_dataset_qa else "custom",
                "context_summary": self._summarize_context(context),
                "generation_metadata": {
                    "strategy": result["strategy"],
                    "model_used": result["model_used"],
                    "confidence": result["confidence"],
                    "context_used": result.get("context_used", [])
                },
                "provenance": {
                    "scene_id": context.get("scene_id"),
                    "keyframe_id": context.get("keyframe_id"),
                    "retrieval_timestamp": context.get("timestamp"),
                    "pipeline_version": "1.0"
                }
            }
            
            # Add dataset-specific information if in dataset mode
            if use_dataset_qa:
                complete_response["dataset_info"] = {
                    "qa_type": qa_type,
                    "qa_serial": qa_serial,
                    "ground_truth": ground_truth,
                    "evaluation": self.evaluator.evaluate_answer(
                        result["answer"], ground_truth, qa_type
                    )
                }
            
            # Add detailed analysis for hybrid results
            if result["strategy"] == "hybrid":
                complete_response["detailed_analysis"] = {
                    "visual_analysis": result.get("visual_analysis"),
                    "structured_analysis": result.get("structured_analysis")
                }
            
            logger.info(f"Successfully answered query using {result['strategy']} strategy")
            return complete_response
            
        except Exception as e:
            logger.error(f"RAG pipeline failed: {e}")
            self.stats["failed_generations"] += 1
            return {
                "answer": f"Pipeline error: {e}",
                "success": False,
                "error": str(e),
                "pipeline_stage": "orchestration"
            }
    
    def batch_answer_questions(self, queries: list, **kwargs) -> list:
        """
        Answer multiple questions in batch.
        
        Args:
            queries: List of query strings
            **kwargs: Arguments passed to answer_question
            
        Returns:
            List of answer dictionaries
        """
        logger.info(f"Processing batch of {len(queries)} queries")
        
        results = []
        for i, query in enumerate(queries):
            logger.info(f"Processing query {i+1}/{len(queries)}")
            result = self.answer_question(query, **kwargs)
            results.append(result)
        
        logger.info(f"Batch processing complete: {len(results)} results")
        return results
    
    def _parse_strategy(self, force_strategy: Optional[str]) -> Optional[bool]:
        """Parse strategy string to boolean for generation pipeline"""
        if force_strategy == "multimodal":
            return True
        elif force_strategy == "text_only":
            return False
        else:
            return None  # Auto-determine
    
    def _summarize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of retrieved context for response metadata.
        
        Args:
            context: Retrieved context dictionary
            
        Returns:
            Context summary
        """
        summary = {
            "has_visual_context": context.get("image_context") is not None,
            "has_structured_context": context.get("structured_context") is not None,
            "has_text_context": context.get("text_context") is not None,
            "query_analysis": context.get("query_analysis", {}).get("query_type", "unknown")
        }
        
        # Add specific counts if available
        if context.get("structured_context"):
            structured = context["structured_context"]
            summary["vehicle_data_available"] = "vehicle_data" in structured
            summary["sensor_data_available"] = "sensor_data" in structured
        
        return summary
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get pipeline usage statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = self.stats.copy()
        stats["model_memory_usage"] = self.model_loader.get_memory_usage()
        stats["generation_stats"] = self.generator.get_generation_stats()
        
        # Calculate success rate
        total_attempts = stats["successful_generations"] + stats["failed_generations"]
        if total_attempts > 0:
            stats["success_rate"] = stats["successful_generations"] / total_attempts
        else:
            stats["success_rate"] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reset pipeline statistics"""
        self.stats = {
            "queries_processed": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "multimodal_queries": 0,
            "text_only_queries": 0
        }
        logger.info("Pipeline statistics reset")
    
    def unload_models(self):
        """Unload all models to free memory"""
        self.model_loader.unload_models()
        logger.info("All models unloaded from RAG pipeline")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on pipeline components.
        
        Returns:
            Health status dictionary
        """
        health = {
            "pipeline_status": "healthy",
            "retriever_status": "healthy",  # TODO: Implement actual health checks
            "generator_status": "healthy",
            "model_loader_status": "healthy",
            "memory_usage": self.model_loader.get_memory_usage(),
            "stats": self.get_pipeline_stats()
        }
        
        return health
    
    def answer_dataset_qa(
        self,
        scene_id: int,
        keyframe_id: int, 
        qa_type: str,
        qa_serial: int,
        force_strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convenience method for answering dataset QA pairs.
        
        Args:
            scene_id: Scene ID
            keyframe_id: Keyframe ID  
            qa_type: QA type ("perception", "planning", "prediction", "behavior")
            qa_serial: QA pair serial number (1-indexed)
            force_strategy: Force generation strategy
            
        Returns:
            Answer with ground truth comparison
        """
        return self.answer_question(
            use_dataset_qa=True,
            scene_id=scene_id,
            keyframe_id=keyframe_id,
            qa_type=qa_type,
            qa_serial=qa_serial,
            force_strategy=force_strategy
        )
    
    def evaluate_dataset_qa_batch(
        self,
        qa_specifications: List[Dict[str, Any]],
        force_strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate multiple dataset QA pairs in batch.
        
        Args:
            qa_specifications: List of dicts with keys: scene_id, keyframe_id, qa_type, qa_serial
            force_strategy: Force generation strategy for all
            
        Returns:
            Batch evaluation results with aggregated metrics
        """
        logger.info(f"Evaluating batch of {len(qa_specifications)} dataset QA pairs")
        
        results = []
        evaluation_metrics = {
            "total_pairs": len(qa_specifications),
            "successful_answers": 0,
            "failed_answers": 0,
            "accuracy_scores": [],
            "semantic_scores": [],
            "by_qa_type": {}
        }
        
        for i, spec in enumerate(qa_specifications):
            logger.info(f"Processing QA pair {i+1}/{len(qa_specifications)}")
            
            result = self.answer_dataset_qa(
                scene_id=spec["scene_id"],
                keyframe_id=spec["keyframe_id"],
                qa_type=spec["qa_type"],
                qa_serial=spec["qa_serial"],
                force_strategy=force_strategy
            )
            
            results.append(result)
            
            # Aggregate metrics
            if result["success"]:
                evaluation_metrics["successful_answers"] += 1
                eval_data = result["dataset_info"]["evaluation"]
                evaluation_metrics["accuracy_scores"].append(eval_data["overall_score"])
                evaluation_metrics["semantic_scores"].append(eval_data["semantic_similarity"])
                
                # Track by QA type
                qa_type = spec["qa_type"]
                if qa_type not in evaluation_metrics["by_qa_type"]:
                    evaluation_metrics["by_qa_type"][qa_type] = {
                        "count": 0,
                        "accuracy_scores": [],
                        "semantic_scores": []
                    }
                evaluation_metrics["by_qa_type"][qa_type]["count"] += 1
                evaluation_metrics["by_qa_type"][qa_type]["accuracy_scores"].append(eval_data["overall_score"])
                evaluation_metrics["by_qa_type"][qa_type]["semantic_scores"].append(eval_data["semantic_similarity"])
            else:
                evaluation_metrics["failed_answers"] += 1
        
        # Calculate aggregated metrics
        if evaluation_metrics["accuracy_scores"]:
            evaluation_metrics["avg_overall_score"] = sum(evaluation_metrics["accuracy_scores"]) / len(evaluation_metrics["accuracy_scores"])
            evaluation_metrics["avg_semantic_similarity"] = sum(evaluation_metrics["semantic_scores"]) / len(evaluation_metrics["semantic_scores"])
        else:
            evaluation_metrics["avg_overall_score"] = 0.0
            evaluation_metrics["avg_semantic_similarity"] = 0.0
        
        # Calculate per-QA-type metrics
        for qa_type, type_data in evaluation_metrics["by_qa_type"].items():
            if type_data["accuracy_scores"]:
                type_data["avg_overall_score"] = sum(type_data["accuracy_scores"]) / len(type_data["accuracy_scores"])
                type_data["avg_semantic_similarity"] = sum(type_data["semantic_scores"]) / len(type_data["semantic_scores"])
        
        evaluation_metrics["success_rate"] = evaluation_metrics["successful_answers"] / evaluation_metrics["total_pairs"]
        
        return {
            "results": results,
            "evaluation_metrics": evaluation_metrics,
            "pipeline_stats": self.get_pipeline_stats()
        }
    
