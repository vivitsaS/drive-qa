import google.generativeai as genai
from rag.retrieval.context_retriever import ContextRetriever
import json
import base64
from typing import Dict, Any, Optional
from loguru import logger
import os
api_key = os.getenv("GEMINI_API_KEY")

class RAGAgent:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize RAG Agent with Gemini Flash model and context retriever tools.
        
        Args:
            api_key: Google Gemini API key (optional, will use GEMINI_API_KEY env var if not provided)
        """
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("API key not provided and GEMINI_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.context_retriever = None
        
    def _get_context_tool(self, scene_id: str, keyframe_id: int) -> Dict[str, Any]:
        """Tool to get context data for a specific keyframe."""
        try:
            self.context_retriever = ContextRetriever(scene_id, keyframe_id)
            context_data = self.context_retriever.get_context_for_keyframe_only()
            return {
                "success": True,
                "data": {
                    "scene_id": scene_id,
                    "keyframe_id": keyframe_id,
                    "keyframe_token": self.context_retriever.keyframe_token,
                    "samples_count": len(context_data.get("samples", {})),
                    "keyframes_count": len(context_data.get("key_frames", {}))
                }
            }
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_vehicle_data_tool(self) -> Dict[str, Any]:
        """Tool to get vehicle movement data up to the keyframe."""
        if not self.context_retriever:
            return {"success": False, "error": "Context retriever not initialized"}
        
        try:
            vehicle_data = self.context_retriever.get_vehicle_data_upto_sample_token()
            return {"success": True, "data": vehicle_data}
        except Exception as e:
            logger.error(f"Error getting vehicle data: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_sensor_data_tool(self) -> Dict[str, Any]:
        """Tool to get sensor detection data for the keyframe."""
        if not self.context_retriever:
            return {"success": False, "error": "Context retriever not initialized"}
        
        try:
            sensor_data = self.context_retriever.get_sensor_data_upto_sample_token()
            return {"success": True, "data": sensor_data}
        except Exception as e:
            logger.error(f"Error getting sensor data: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_annotated_images_tool(self) -> Dict[str, Any]:
        """Tool to get annotated images for the keyframe."""
        if not self.context_retriever:
            return {"success": False, "error": "Context retriever not initialized"}
        
        try:
            image_bytes = self.context_retriever.get_annotated_images()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            return {
                "success": True, 
                "data": {
                    "image_base64": image_base64,
                    "format": "png",
                    "description": "Annotated images showing detected objects with bounding boxes"
                }
            }
        except Exception as e:
            logger.error(f"Error getting annotated images: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_raw_images_tool(self) -> Dict[str, Any]:
        """Tool to get raw camera images for the keyframe."""
        if not self.context_retriever:
            return {"success": False, "error": "Context retriever not initialized"}
        
        try:
            # Get raw images from context retriever
            raw_images = self.context_retriever.get_raw_images()
            return {
                "success": True,
                "data": raw_images
            }
        except Exception as e:
            logger.error(f"Error getting raw images: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_qa_pair_tool(self, qa_type: str, qa_serial: int) -> Dict[str, Any]:
        """Tool to get the specific QA pair from annotations."""
        if not self.context_retriever:
            return {"success": False, "error": "Context retriever not initialized"}
        
        try:
            qa_pair = self.context_retriever.get_qa_pair(qa_type, qa_serial)
            if qa_pair:
                return {"success": True, "data": qa_pair}
            else:
                return {"success": False, "error": "QA pair not found"}
        except Exception as e:
            logger.error(f"Error getting QA pair: {e}")
            return {"success": False, "error": str(e)}
    
    def answer_question(self, scene_id: str, keyframe_id: int, qa_type: str, qa_serial: int) -> Dict[str, Any]:
        """
        Answer a question using the RAG system.
        
        Args:
            scene_id: Scene identifier (1-based)
            keyframe_id: Keyframe identifier (1-based)
            qa_type: Type of QA (e.g., 'perception', 'prediction', etc.)
            qa_serial: QA pair serial number (1-based)
            
        Returns:
            Dictionary containing the answer and metadata
        """
        try:
            # Initialize context retriever
            self.context_retriever = ContextRetriever(scene_id, keyframe_id)
            
            # Get the question from annotations
            qa_pair = self.context_retriever.get_qa_pair(qa_type, qa_serial)
            if not qa_pair:
                return {
                    "success": False,
                    "error": f"QA pair not found for type={qa_type}, serial={qa_serial}"
                }
            
            question = qa_pair["Q"]
            ground_truth_answer = qa_pair["A"]
            
            # Prepare tools for the model
            tools = [
                {
                    "name": "get_context",
                    "description": "Get context data for the current scene and keyframe",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "scene_id": {"type": "string"},
                            "keyframe_id": {"type": "integer"}
                        },
                        "required": ["scene_id", "keyframe_id"]
                    }
                },
                {
                    "name": "get_vehicle_data",
                    "description": "Get vehicle movement data up to the current keyframe",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                },
                {
                    "name": "get_sensor_data",
                    "description": "Get sensor detection data for the current keyframe",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                },
                {
                    "name": "get_annotated_images",
                    "description": "Get annotated images showing detected objects",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            ]
            key_objects = self.context_retriever.get_key_objects_in_keyframe()
            
            # Create the prompt
            prompt = f"""
You are an autonomous driving assistant analyzing a driving scene from the nuScenes dataset. Answer the following question about the driving scene:

Question: {question}

IMPORTANT CONTEXT:
- The "ego vehicle" is the autonomous vehicle you're analyzing from - it's equipped with 6 cameras (front, front-left, front-right, back, back-left, back-right) and other sensors
- The ego vehicle is driving in its designated lane - objects detected on the sides of the road (shoulders, medians, adjacent lanes) are NOT in the ego vehicle's path
- Lane markings: Solid white lines typically mark road edges/shoulders; dashed white lines separate lanes going the same direction
- Traffic cones, barriers, or objects on the shoulder/median are NOT blocking the ego vehicle's path unless they extend into the driving lane
- The ego vehicle follows standard driving rules and stays within its lane unless changing lanes most of the times, but anamolous behaviour is possible. This would be categorized as risky driving.

You have access to:
1. Context data about the scene and keyframe
2. Vehicle movement data (speed, acceleration, position, etc.)
3. Sensor data (object detections, LiDAR/radar points, etc.)
4. Annotated images showing detected objects from the ego vehicle's perspective

Focus on:
 - Key objects in the scene: {key_objects}
 - Their locations relative to the ego vehicle's driving path (not just detected anywhere)
 - Whether objects are actually in the ego vehicle's lane vs. adjacent lanes/shoulders
 - Object states (moving, stationary, etc.)
- Traffic conditions and road layout

Provide a concise, accurate answer based on the available data. Consider lane positioning and driving context carefully.
"""
            
            # Get context data first
            context_result = self._get_context_tool(scene_id, keyframe_id)
            if not context_result["success"]:
                return context_result
            
            # Get vehicle data
            vehicle_result = self._get_vehicle_data_tool()
            
            # Get sensor data
            sensor_result = self._get_sensor_data_tool()
            
            # Get annotated images
            images_result = self._get_annotated_images_tool()
            # Save the image to a file (for debugging)
            if images_result["success"]:
                with open("annotated_image.png", "wb") as f:
                    f.write(base64.b64decode(images_result["data"]["image_base64"]))
                logger.info("Image saved to annotated_image.png")
            # logger.info(f"Images result: {images_result}")
            
            # Prepare content for the model
            content_parts = [prompt]
            
            # Add context information
            if context_result["success"]:
                content_parts.append(f"Context: {json.dumps(context_result['data'], indent=2)}")
            
            # Add vehicle data
            if vehicle_result["success"]:
                content_parts.append(f"Vehicle Data: {json.dumps(vehicle_result['data'], indent=2)}")
            
            # Add sensor data
            if sensor_result["success"]:
                content_parts.append(f"Sensor Data: {json.dumps(sensor_result['data'], indent=2)}")
            
            # Add images if available
            if images_result["success"]:
                # Create image part for Gemini
                image_part = {
                    "mime_type": "image/png",
                    "data": images_result["data"]["image_base64"]
                }
                content_parts.append(image_part)
                logger.info(f"Added image to content_parts with base64 length: {len(images_result['data']['image_base64'])}")
                logger.info(f"Image part structure: {type(image_part)} with keys: {list(image_part.keys())}")
            
            # Debug: Log content parts structure
            logger.info(f"Content parts count: {len(content_parts)}")
            for i, part in enumerate(content_parts):
                if isinstance(part, dict) and "mime_type" in part:
                    logger.info(f"Part {i}: Image with mime_type={part['mime_type']}, data_length={len(part['data'])}")
                else:
                    logger.info(f"Part {i}: Text with length={len(str(part))}")
            
            # Generate response
            response = self.model.generate_content(content_parts)
            
            return {
                "success": True,
                "question": question,
                "model_answer": response.text,
                "ground_truth_answer": ground_truth_answer,
                "metadata": {
                    "scene_id": scene_id,
                    "keyframe_id": keyframe_id,
                    "qa_type": qa_type,
                    "qa_serial": qa_serial,
                    "context_available": context_result["success"],
                    "vehicle_data_available": vehicle_result["success"],
                    "sensor_data_available": sensor_result["success"],
                    "images_available": images_result["success"]
                }
            }
            
        except Exception as e:
            logger.error(f"Error in answer_question: {e}")
            return {
                "success": False,
                "error": str(e),
                "metadata": {
                    "scene_id": scene_id,
                    "keyframe_id": keyframe_id,
                    "qa_type": qa_type,
                    "qa_serial": qa_serial
                }
            }


# Example usage
if __name__ == "__main__":
    # The API key should be set as an environment variable
    # export GEMINI_API_KEY="your_gemini_api_key_here"
    
    agent = RAGAgent()
    
    # Example call
    result = agent.answer_question(
        scene_id="1",
        keyframe_id=1,
        qa_type="planning",
        qa_serial=1
    )
    
    if result["success"]:
        print(f"Question: {result['question']}")
        print(f"Model Answer: {result['model_answer']}")
        print(f"Ground Truth: {result['ground_truth_answer']}")
    else:
        print(f"Error: {result['error']}") 