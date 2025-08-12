"""
Query Processor

Analyzes user queries to determine what type of context needs to be retrieved.
"""

import re
from typing import Dict, List, Any
from loguru import logger


class QueryProcessor:
    """Process and analyze user queries for context retrieval"""
    
    def __init__(self):
        """Initialize query processor with intent patterns"""
        self.visual_keywords = [
            "see", "look", "visible", "appear", "show", "display", "image", "picture",
            "front", "behind", "left", "right", "color", "shape", "size"
        ]
        
        self.vehicle_keywords = [
            "speed", "velocity", "acceleration", "brake", "braking", "turn", "turning",
            "stop", "stopping", "slow", "fast", "direction", "heading", "movement"
        ]
        
        self.sensor_keywords = [
            "detect", "sensor", "lidar", "radar", "camera", "object", "obstacle",
            "car", "vehicle", "pedestrian", "bike", "truck", "traffic", "signal"
        ]
        
        self.temporal_keywords = [
            "before", "after", "during", "when", "while", "previous", "next",
            "sequence", "timeline", "history", "future", "past"
        ]
        
        self.qa_type_keywords = {
            "perception": ["see", "detect", "identify", "recognize", "visible", "object"],
            "planning": ["should", "plan", "route", "path", "navigate", "go", "drive"],
            "prediction": ["will", "going", "predict", "future", "next", "likely"],
            "behavior": ["why", "how", "reason", "cause", "behavior", "action"]
        }
        
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to determine context requirements.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with query analysis results
        """
        query_lower = query.lower()
        
        analysis = {
            "original_query": query,
            "needs_visual_context": self._needs_visual_context(query_lower),
            "needs_vehicle_data": self._needs_vehicle_data(query_lower),
            "needs_sensor_data": self._needs_sensor_data(query_lower),
            "needs_temporal_context": self._needs_temporal_context(query_lower),
            "needs_qa_context": self._needs_qa_context(query_lower),
            "needs_object_detection": self._needs_object_detection(query_lower),
            "needs_structured_context": True,  # Most queries need some structured context
            
            # Specific extractions
            "relevant_qa_types": self._extract_qa_types(query_lower),
            "relevant_cameras": self._extract_cameras(query_lower),
            "objects_of_interest": self._extract_objects(query_lower),
            "query_type": self._classify_query_type(query_lower),
            "intent": self._extract_intent(query_lower)
        }
        
        logger.info(f"Query analysis: {analysis['query_type']} query about {analysis['intent']}")
        return analysis
    
    def _needs_visual_context(self, query: str) -> bool:
        """Check if query requires visual/image context"""
        return any(keyword in query for keyword in self.visual_keywords)
    
    def _needs_vehicle_data(self, query: str) -> bool:
        """Check if query requires vehicle state data"""
        return any(keyword in query for keyword in self.vehicle_keywords)
    
    def _needs_sensor_data(self, query: str) -> bool:
        """Check if query requires sensor/detection data"""
        return any(keyword in query for keyword in self.sensor_keywords)
    
    def _needs_temporal_context(self, query: str) -> bool:
        """Check if query requires temporal context"""
        return any(keyword in query for keyword in self.temporal_keywords)
    
    def _needs_qa_context(self, query: str) -> bool:
        """Check if query would benefit from existing QA pairs"""
        # Most queries can benefit from QA context for examples
        return True
    
    def _needs_object_detection(self, query: str) -> bool:
        """Check if query requires object detection/annotation"""
        object_keywords = ["object", "car", "vehicle", "pedestrian", "bike", "truck", "detect"]
        return any(keyword in query for keyword in object_keywords)
    
    def _extract_qa_types(self, query: str) -> List[str]:
        """Extract relevant QA types for the query"""
        relevant_types = []
        
        for qa_type, keywords in self.qa_type_keywords.items():
            if any(keyword in query for keyword in keywords):
                relevant_types.append(qa_type)
        
        # Default to perception if no specific type detected
        if not relevant_types:
            relevant_types = ["perception"]
        
        return relevant_types
    
    def _extract_cameras(self, query: str) -> List[str]:
        """Extract relevant camera views from query"""
        camera_mapping = {
            "front": ["CAM_FRONT"],
            "back": ["CAM_BACK"],
            "left": ["CAM_FRONT_LEFT", "CAM_BACK_LEFT"],
            "right": ["CAM_FRONT_RIGHT", "CAM_BACK_RIGHT"],
            "side": ["CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
        }
        
        relevant_cameras = []
        for direction, cameras in camera_mapping.items():
            if direction in query:
                relevant_cameras.extend(cameras)
        
        # Default to front camera if no specific direction mentioned
        if not relevant_cameras:
            relevant_cameras = ["CAM_FRONT"]
        
        return list(set(relevant_cameras))  # Remove duplicates
    
    def _extract_objects(self, query: str) -> List[str]:
        """Extract objects of interest from query"""
        object_keywords = {
            "car": ["car", "vehicle", "automobile"],
            "pedestrian": ["pedestrian", "person", "walker"],
            "bicycle": ["bike", "bicycle", "cyclist"],
            "truck": ["truck", "lorry"],
            "traffic_sign": ["sign", "signal", "light"],
            "barrier": ["barrier", "cone", "obstacle"]
        }
        
        objects_of_interest = []
        for obj_type, keywords in object_keywords.items():
            if any(keyword in query for keyword in keywords):
                objects_of_interest.append(obj_type)
        
        return objects_of_interest
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the overall type of query"""
        if any(word in query for word in ["what", "which", "identify"]):
            return "identification"
        elif any(word in query for word in ["how", "why", "explain"]):
            return "explanation"
        elif any(word in query for word in ["is", "are", "can", "will"]):
            return "yes_no"
        elif any(word in query for word in ["where", "locate"]):
            return "location"
        elif any(word in query for word in ["when", "time"]):
            return "temporal"
        else:
            return "general"
    
    def _extract_intent(self, query: str) -> str:
        """Extract the main intent/topic of the query"""
        # TODO: Implement more sophisticated intent extraction
        if "brake" in query or "braking" in query:
            return "braking_behavior"
        elif "speed" in query or "velocity" in query:
            return "vehicle_speed"
        elif "object" in query or "detect" in query:
            return "object_detection"
        elif "turn" in query or "direction" in query:
            return "vehicle_direction"
        else:
            return "general_driving"