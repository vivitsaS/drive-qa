"""
Generation Pipeline

Handles the generation phase of RAG, coordinating between LLaVA and Llama 3 models
to produce coherent answers based on retrieved context.
"""

from typing import Dict, Any, Optional
from loguru import logger

from ..models.model_loader import ModelLoader
from ..models.llava_model import LLaVAModel
from ..models.llama_model import LlamaModel


class GenerationPipeline:
    """Generation pipeline for RAG system"""
    
    def __init__(self, model_loader: ModelLoader):
        """
        Initialize generation pipeline.
        
        Args:
            model_loader: Initialized ModelLoader instance
        """
        self.model_loader = model_loader
        self._llava_model: Optional[LLaVAModel] = None
        self._llama_model: Optional[LlamaModel] = None
        
    def generate_answer(
        self, 
        query: str, 
        context: Dict[str, Any],
        use_multimodal: bool = None
    ) -> Dict[str, Any]:
        """
        Generate answer based on query and retrieved context.
        
        Args:
            query: Original user query
            context: Retrieved context from ContextualRetriever
            use_multimodal: Force multimodal (True) or text-only (False), auto if None
            
        Returns:
            Dictionary containing generated answer and metadata
        """
        try:
            logger.info(f"Generating answer for query: {query}")
            
            # Determine generation strategy
            generation_strategy = self._determine_strategy(context, use_multimodal)
            
            if generation_strategy == "multimodal":
                return self._generate_multimodal_answer(query, context)
            elif generation_strategy == "text_only":
                return self._generate_text_answer(query, context)
            elif generation_strategy == "hybrid":
                return self._generate_hybrid_answer(query, context)
            else:
                raise ValueError(f"Unknown generation strategy: {generation_strategy}")
                
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return {
                "answer": f"Error generating answer: {e}",
                "confidence": 0.0,
                "strategy": "error",
                "error": str(e)
            }
    
    def _determine_strategy(self, context: Dict[str, Any], force_strategy: Optional[bool]) -> str:
        """
        Determine the best generation strategy based on available context.
        
        Args:
            context: Retrieved context
            force_strategy: Force multimodal (True) or text-only (False)
            
        Returns:
            Strategy name: "multimodal", "text_only", or "hybrid"
        """
        has_images = context.get("image_context") is not None
        has_structured = context.get("structured_context") is not None
        
        if force_strategy is True:
            return "multimodal"
        elif force_strategy is False:
            return "text_only"
        else:
            # Auto-determine strategy
            if has_images and has_structured:
                return "hybrid"
            elif has_images:
                return "multimodal"
            else:
                return "text_only"
    
    def _generate_multimodal_answer(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate answer using LLaVA for multimodal reasoning.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Generated answer with metadata
        """
        logger.info("Using multimodal generation (LLaVA)")
        
        # Load LLaVA model if needed
        if self._llava_model is None:
            self._llava_model = self.model_loader.load_llava()
        
        # Format prompt for LLaVA
        prompt = self._format_multimodal_prompt(query, context)
        
        # Extract image from context
        image = self._extract_image_from_context(context)
        
        # Generate response
        response = self._llava_model.generate_response(prompt, image)
        
        return {
            "answer": response,
            "strategy": "multimodal",
            "model_used": "LLaVA",
            "confidence": 0.8,  # TODO: Implement confidence estimation
            "context_used": ["image", "structured"]
        }
    
    def _generate_text_answer(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate answer using Llama 3 for text-only reasoning.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Generated answer with metadata
        """
        logger.info("Using text-only generation (Llama 3)")
        
        # Load Llama 3 model if needed
        if self._llama_model is None:
            self._llama_model = self.model_loader.load_llama()
        
        # Use Llama 3's structured data analysis
        structured_data = context.get("structured_context", {})
        response = self._llama_model.analyze_structured_data(structured_data, query)
        
        return {
            "answer": response,
            "strategy": "text_only",
            "model_used": "Llama 3",
            "confidence": 0.7,  # TODO: Implement confidence estimation
            "context_used": ["structured", "text"]
        }
    
    def _generate_hybrid_answer(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate answer using both models and combine results.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Combined answer with metadata
        """
        logger.info("Using hybrid generation (LLaVA + Llama 3)")
        
        # Get visual analysis from LLaVA
        visual_analysis = self._generate_multimodal_answer(query, context)
        
        # Get structured analysis from Llama 3
        text_analysis = self._generate_text_answer(query, context)
        
        # Combine results
        combined_answer = self._combine_answers(visual_analysis, text_analysis, query)
        
        return {
            "answer": combined_answer,
            "strategy": "hybrid",
            "model_used": "LLaVA + Llama 3",
            "confidence": 0.9,  # Higher confidence from combined analysis
            "context_used": ["image", "structured", "text"],
            "visual_analysis": visual_analysis["answer"],
            "structured_analysis": text_analysis["answer"]
        }
    
    def _format_multimodal_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """
        Format prompt for LLaVA multimodal reasoning.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Formatted prompt
        """
        base_prompt = f"""You are an expert at analyzing autonomous driving scenarios. 
        
        Look at the provided camera image and answer the following question based on what you see and the additional context provided.
        
        Question: {query}
        
        Additional Context:"""
        
        # Add structured context if available
        if context.get("structured_context"):
            structured = context["structured_context"]
            if "vehicle_data" in structured:
                base_prompt += f"\nVehicle Data: {structured['vehicle_data']}"
            if "sensor_data" in structured:
                base_prompt += f"\nSensor Data: {structured['sensor_data']}"
        
        base_prompt += "\n\nProvide a clear, factual answer based on the visual evidence and context."
        
        return base_prompt
    
    def _combine_answers(
        self, 
        visual_analysis: Dict[str, Any], 
        text_analysis: Dict[str, Any], 
        query: str
    ) -> str:
        """
        Combine visual and text analysis into coherent answer.
        
        Args:
            visual_analysis: Result from LLaVA
            text_analysis: Result from Llama 3
            query: Original query
            
        Returns:
            Combined answer
        """
        # TODO: Implement intelligent answer combination
        # For now, simple concatenation with analysis
        combined = f"""Based on combined visual and data analysis:

        Visual Analysis: {visual_analysis['answer']}
        
        Data Analysis: {text_analysis['answer']}
        
        Conclusion: [TODO: Implement intelligent synthesis]"""
        
        return combined
    
    def _extract_image_from_context(self, context: Dict[str, Any]):
        """
        Extract image data from context for LLaVA processing.
        
        Args:
            context: Retrieved context dictionary
            
        Returns:
            Image data or None
        """
        try:
            # Check if image context is available
            image_context = context.get("image_context")
            if not image_context:
                return None
            
            # Try to get annotated images from context
            annotated_images = image_context.get("annotated_images")
            if annotated_images and annotated_images != "TODO: Get annotated images":
                return annotated_images
            
            # Fallback: try to get images from scene data through context retriever
            scene_id = context.get("scene_id")
            keyframe_id = context.get("keyframe_id")
            
            if scene_id and keyframe_id:
                from ..context_retriever import ContextRetriever
                ctx_retriever = ContextRetriever(scene_id, keyframe_id)
                try:
                    # Get annotated images as bytes
                    image_bytes = ctx_retriever.get_annotated_images()
                    return image_bytes
                except Exception as e:
                    logger.warning(f"Failed to get annotated images: {e}")
                    return None
                    
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract image from context: {e}")
            return None
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about generation pipeline usage.
        
        Returns:
            Statistics dictionary
        """
        return {
            "llava_loaded": self._llava_model is not None,
            "llama_loaded": self._llama_model is not None,
            "total_memory_usage": self.model_loader.get_memory_usage()
        }