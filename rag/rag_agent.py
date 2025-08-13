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
                    "keyframes_count": len(context_data.get("key_frames", {})),
                    "data": context_data
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
    
    # def _get_raw_images_tool(self) -> Dict[str, Any]:
    #     """Tool to get raw camera images for the keyframe."""
    #     if not self.context_retriever:
    #         return {"success": False, "error": "Context retriever not initialized"}
        
    #     try:
    #         # Get raw images from context retriever
    #         raw_images = self.context_retriever.get_raw_images()
    #         return {
    #             "success": True,
    #             "data": raw_images
    #         }
    #     except Exception as e:
    #         logger.error(f"Error getting raw images: {e}")
    #         return {"success": False, "error": str(e)}
    
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
    
    def _execute_tool_call(self, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call based on the function name and arguments."""
        try:
            if function_name == "get_context":
                scene_id = arguments.get("scene_id")
                keyframe_id = arguments.get("keyframe_id")
                return self._get_context_tool(scene_id, keyframe_id)
            elif function_name == "get_vehicle_data":
                return self._get_vehicle_data_tool()
            elif function_name == "get_sensor_data":
                return self._get_sensor_data_tool()
            elif function_name == "get_annotated_images":
                return self._get_annotated_images_tool()
            # elif function_name == "get_raw_images":
            #     return self._get_raw_images_tool()
            else:
                return {"success": False, "error": f"Unknown function: {function_name}"}
        except Exception as e:
            logger.error(f"Error executing tool call {function_name}: {e}")
            return {"success": False, "error": str(e)}
    
    def answer_question(self, scene_id, keyframe_id: int, qa_type: str, qa_serial: int) -> Dict[str, Any]:
        """
        Answer a question using the RAG system with a two-step approach:
        1. First call: Determine what data is needed using tool calls
        2. Second call: Answer the question using the retrieved data
        
        Args:
            scene_id: Scene identifier (1-based) - can be string or integer
            keyframe_id: Keyframe identifier (1-based)
            qa_type: Type of QA (e.g., 'perception', 'prediction', etc.)
            qa_serial: QA pair serial number (1-based)
            
        Returns:
            Dictionary containing the answer and metadata
        """
        try:
            # Convert scene_id to string if it's an integer
            scene_id_str = str(scene_id)
            logger.info(f"Initializing with scene_id_str: {scene_id_str}, keyframe_id: {keyframe_id}")
            
            # Initialize context retriever
            # Pass the original scene_id (could be int or string) to ContextRetriever
            # ContextRetriever will handle the conversion internally
            self.context_retriever = ContextRetriever(scene_id, keyframe_id)
            logger.info("ContextRetriever initialized successfully")
            
            # Get the question from annotations
            logger.info(f"Getting QA pair for type={qa_type}, serial={qa_serial}")
            qa_pair = self.context_retriever.get_qa_pair(qa_type, qa_serial)
            logger.info(f"QA pair retrieved: {qa_pair is not None}")
            if not qa_pair:
                return {
                    "success": False,
                    "error": f"QA pair not found for type={qa_type}, serial={qa_serial}"
                }
            
            question = qa_pair["Q"]
            ground_truth_answer = qa_pair["A"]
            logger.info(f"Question: {question}")
            
            # Prepare tools for the model
            logger.info("Preparing tools for the model")
            tools = [
                {
                    "function_declarations": [
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
                            "description": "Get annotated images showing detected objects. It will have 6 images- from all 6 view points: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "required": []
                            }
                        }
                    ]
                }
            ]
            logger.info("Tools prepared successfully")
            
            # key_objects = self.context_retriever.get_key_objects_in_keyframe()
            
            # STEP 1: First call to determine what data is needed
            first_prompt = f"""
You are an autonomous driving assistant. You need to answer the following question about a driving scene:

Question: {question}

You have access to the following tools to gather information:
1. get_context - Get context data about the scene and keyframe
2. get_vehicle_data - Get vehicle movement data (speed, acceleration, position, etc.)
3. get_sensor_data - Get sensor data (object detections, LiDAR/radar points, etc.)
4. get_annotated_images - Get annotated images showing detected objects. It will have 6 images- from all 6 view points: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT

Call ALL the tools that would be helpful for answering this question. For example:
- If the question is about objects in the scene, call get_sensor_data and get_annotated_images
- If the question is about vehicle behavior, call get_vehicle_data and get_context
- If the question is about the overall scene, call get_context and get_annotated_images

Only generate tool calls, don't generate any other text.

"""
            
            # Create a model with tools enabled
            logger.info("Creating model with tools")
            model_with_tools = genai.GenerativeModel('gemini-1.5-flash', tools=tools)
            logger.info("Model created successfully")
            
            # First call to get tool calls
            logger.info("Making first call to model")
            first_response = model_with_tools.generate_content(first_prompt)
            logger.info("First response received")
            
            # Extract tool calls and execute them
            retrieved_data = {}
            tool_call_count = 0
            logger.info(f"First response type: {type(first_response)}")
            logger.info(f"First response has candidates: {hasattr(first_response, 'candidates')}")
            
            if hasattr(first_response, 'candidates') and first_response.candidates:
                logger.info(f"Number of candidates: {len(first_response.candidates)}")
                candidate = first_response.candidates[0]
                logger.info(f"Candidate type: {type(candidate)}")
                logger.info(f"Candidate has content: {hasattr(candidate, 'content')}")
                
                if hasattr(candidate, 'content') and candidate.content:
                    logger.info(f"Content type: {type(candidate.content)}")
                    logger.info(f"Content has parts: {hasattr(candidate.content, 'parts')}")
                    
                    for part in candidate.content.parts:
                        logger.info(f"Part type: {type(part)}")
                        logger.info(f"Part has function_call: {hasattr(part, 'function_call')}")
                        
                        if hasattr(part, 'function_call') and part.function_call:
                            tool_call_count += 1
                            function_call = part.function_call
                            function_name = function_call.name
                            arguments = json.loads(function_call.args) if function_call.args else {}
                            
                            # Ensure scene_id is passed as string for get_context tool
                            if function_name == "get_context" and "scene_id" in arguments:
                                arguments["scene_id"] = scene_id_str
                            
                            logger.info(f"Executing tool call {tool_call_count}: {function_name} with args: {arguments}")
                            result = self._execute_tool_call(function_name, arguments)
                            retrieved_data[function_name] = result
            
            logger.info(f"Total tool calls executed: {tool_call_count}")
            logger.info(f"Tools called: {list(retrieved_data.keys())}")
            
            # STEP 2: Second call with the retrieved data
            second_prompt = f"""
You are an autonomous driving assistant. Answer the following question about a driving scene:

Question: {question}

Here is the data needed to answer the question:

"""
            
            # Add retrieved data to the prompt
            content_parts = [second_prompt]
            
            for tool_name, result in retrieved_data.items():
                if result["success"]:
                    if tool_name == "get_annotated_images":
                        # Handle images specially
                        image_part = {
                            "mime_type": "image/png",
                            "data": result["data"]["image_base64"]
                        }
                        content_parts.append(image_part)
                        content_parts.append(f"\nAnnotated Image Data: {result['data']['description']}")
                    else:
                        content_parts.append(f"{tool_name.replace('_', ' ').title()} Data: {json.dumps(result['data'], indent=2)}")
                else:
                    content_parts.append(f"{tool_name.replace('_', ' ').title()} Error: {result['error']}")
            # Add final instructions
            content_parts.append(f"""

Based on the data above, provide a comprehensive answer to the question.

Focus on:
- What objects are present in the scene
- Their locations relative to the ego vehicle and how relevant they are to the question/driving path
- Their states (moving, stationary, etc.)
- Any relevant spatial relationships
- Traffic conditions and road layout

Provide a concise answer based on the available data. One line answer is enough, but don't miss out any important information!
""")
            
            # logger.info(f"Content parts: {content_parts.keys()}")
            # Generate final response
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
                    "tools_called": list(retrieved_data.keys()),
                    "data_retrieved": {k: v["success"] for k, v in retrieved_data.items()}
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
        scene_id=1,
        keyframe_id=1,
        qa_type="planning",
        qa_serial=1
    )
    
    if result["success"]:
        print(f"Question: {result['question']}")
        print(f"Model Answer: {result['model_answer']}")
        print(f"Ground Truth: {result['ground_truth_answer']}")
        print(f"Tools called: {result['metadata']['tools_called']}")
    else:
        print(f"Error: {result['error']}") 