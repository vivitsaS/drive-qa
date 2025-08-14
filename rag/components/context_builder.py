"""
Context Builder Component

Builds context for LLM interactions by combining various data sources.
"""

from typing import Dict, Any, List, Union
from loguru import logger

from analysis.result import AnalysisResult
from .data_retriever import DataRetriever


class ContextBuilder:
    """Builds context for LLM interactions"""
    
    def __init__(self, data_retriever: DataRetriever):
        """
        Initialize context builder.
        
        Args:
            data_retriever: Data retriever instance
        """
        self.data_retriever = data_retriever
    
    def build_context_for_qa(self, scene_id: Union[int, str], keyframe_id: Union[int, str], 
                           qa_type: str, qa_serial: int) -> AnalysisResult:
        """
        Build complete context for answering a specific QA pair.
        
        Args:
            scene_id: Scene identifier
            keyframe_id: Keyframe identifier
            qa_type: Type of QA
            qa_serial: QA pair serial number
            
        Returns:
            AnalysisResult with complete context
        """
        try:
            # Get the QA pair first
            qa_result = self.data_retriever.get_qa_pair(scene_id, keyframe_id, qa_type, qa_serial)
            if not qa_result.status.value == 'success':
                return qa_result
            
            qa_pair = qa_result.data
            question = qa_pair.get("Q", "")
            ground_truth = qa_pair.get("A", "")
            
            # Get keyframe information
            keyframe_result = self.data_retriever.get_keyframe_info(scene_id, keyframe_id)
            key_objects = {}
            if keyframe_result.status.value == 'success':
                key_objects = keyframe_result.data.get('key_objects', {})
            
            # Get vehicle data
            vehicle_result = self.data_retriever.get_vehicle_data_upto_keyframe(scene_id, keyframe_id)
            vehicle_data = {}
            if vehicle_result.status.value == 'success':
                vehicle_data = vehicle_result.data
            
            # Get sensor data
            sensor_result = self.data_retriever.get_sensor_data_for_keyframe(scene_id, keyframe_id)
            sensor_data = {}
            if sensor_result.status.value == 'success':
                sensor_data = sensor_result.data
            
            # Get annotated images
            images_result = self.data_retriever.get_annotated_images(scene_id, keyframe_id)
            image_data = {}
            if images_result.status.value == 'success':
                image_data = images_result.data
            
            # Build the complete context
            context = {
                'question': question,
                'ground_truth': ground_truth,
                'scene_id': scene_id,
                'keyframe_id': keyframe_id,
                'qa_type': qa_type,
                'qa_serial': qa_serial,
                'key_objects': key_objects,
                'vehicle_data': vehicle_data,
                'sensor_data': sensor_data,
                'image_data': image_data,
                'metadata': {
                    'vehicle_data_available': vehicle_result.status.value == 'success',
                    'sensor_data_available': sensor_result.status.value == 'success',
                    'images_available': images_result.status.value == 'success',
                    'keyframe_info_available': keyframe_result.status.value == 'success'
                }
            }
            
            return AnalysisResult.success(
                data=context,
                metadata={
                    'scene_id': scene_id,
                    'keyframe_id': keyframe_id,
                    'qa_type': qa_type,
                    'qa_serial': qa_serial
                }
            )
            
        except Exception as e:
            logger.error(f"Error building context: {e}")
            return AnalysisResult.error(
                str(e),
                metadata={
                    'scene_id': scene_id,
                    'keyframe_id': keyframe_id,
                    'qa_type': qa_type,
                    'qa_serial': qa_serial
                }
            )
    
    def build_prompt(self, context: Dict[str, Any]) -> str:
        """
        Build the prompt for the LLM based on the context.
        
        Args:
            context: Complete context data
            
        Returns:
            Formatted prompt string
        """
        question = context.get('question', '')
        key_objects = context.get('key_objects', {})
        
        # Format key objects for the prompt
        key_objects_str = ""
        if key_objects:
            key_objects_str = f" - Key objects in the scene: {list(key_objects.keys())}"
        
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
 - Key objects in the scene: {key_objects_str}
 - Their locations relative to the ego vehicle's driving path (not just detected anywhere)
 - Whether objects are actually in the ego vehicle's lane vs. adjacent lanes/shoulders
 - Object states (moving, stationary, etc.)
- Traffic conditions and road layout

Provide a concise, accurate answer based on the available data. Consider lane positioning and driving context carefully.
"""
        
        return prompt
    
    def build_content_parts(self, context: Dict[str, Any]) -> List[Any]:
        """
        Build content parts for the LLM including text and images.
        
        Args:
            context: Complete context data
            
        Returns:
            List of content parts (text and images)
        """
        content_parts = []
        
        # Add the prompt
        prompt = self.build_prompt(context)
        content_parts.append(prompt)
        
        # Add context information
        vehicle_data = context.get('vehicle_data', {})
        if vehicle_data:
            import json
            content_parts.append(f"Vehicle Data: {json.dumps(vehicle_data, indent=2)}")
        
        # Add sensor data
        sensor_data = context.get('sensor_data', {})
        if sensor_data:
            import json
            content_parts.append(f"Sensor Data: {json.dumps(sensor_data, indent=2)}")
        
        # Add images if available
        image_data = context.get('image_data', {})
        if image_data and 'image_base64' in image_data:
            image_part = {
                "mime_type": "image/png",
                "data": image_data['image_base64']
            }
            content_parts.append(image_part)
        
        return content_parts 