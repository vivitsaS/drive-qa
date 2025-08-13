from parsers.data_loader import DataLoader
from loguru import logger
import cv2
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from typing import Dict, List, Any


class ContextRetriever:
    def __init__(self, scene_id, keyframe_id):
        self.data_loader = DataLoader()
        self.scene_id = scene_id
        self.keyframe_id = keyframe_id
        self.keyframe_token = self.data_loader._assign_keyframe_token(scene_id, keyframe_id)
        # Cache scene data to avoid reloading
        self._scene_data = None
        self.context_data = self.get_context_upto_keyframe()

    def _sort_keyframes_by_timestamp(self, scene_data):
        """
        Sort keyframe tokens by timestamp based on their position in the samples dict.
        Since samples are already sorted by timestamp, we can use their order to sort keyframes.
        
        Args:
            scene_data: The scene data containing samples and key_frames
            
        Returns:
            list: Sorted keyframe tokens by timestamp
        """
        sample_tokens = list(scene_data['samples'].keys())
        keyframe_tokens = list(scene_data['key_frames'].keys())
        
        # Sort keyframe tokens based on their position in sample_tokens
        # Keyframes that appear earlier in samples come first
        sorted_keyframe_tokens = sorted(keyframe_tokens, 
                                      key=lambda token: sample_tokens.index(token) if token in sample_tokens else float('inf'))
        
        # logger.info(f"Sorted keyframe tokens by timestamp: {sorted_keyframe_tokens}")
        return sorted_keyframe_tokens

    def get_context_for_keyframe_only  (self):
        # Use cached scene data if available, otherwise load it
        if self._scene_data is None:
            self._scene_data = self.data_loader.load_scene_data(self.scene_id)
        scene_data = self._scene_data
        
        # Use cached keyframe token
        keyframe_token = self.keyframe_token
        
        # Get the specific sample and keyframe corresponding to this token
        if keyframe_token not in scene_data['samples']:
            logger.error(f"Keyframe token {keyframe_token} not found in samples")
            return None
            
        if keyframe_token not in scene_data['key_frames']:
            logger.error(f"Keyframe token {keyframe_token} not found in keyframes")
            return None
        
        # Create context with just the specific sample and keyframe
        context_data = scene_data.copy()
        context_data['samples'] = {keyframe_token: scene_data['samples'][keyframe_token]}
        context_data['key_frames'] = {keyframe_token: scene_data['key_frames'][keyframe_token]}
        
        # logger.info(f"Retrieved context for keyframe token: {keyframe_token}")
        
        return context_data
    
    def get_key_objects_in_keyframe(self):
        keyframe_data = self.get_context_for_keyframe_only()["key_frames"][self.keyframe_token]["key_object_infos"]
        return keyframe_data


    def get_context_upto_keyframe(self):
        # get scene data - cache it for reuse
        if self._scene_data is None:
            self._scene_data = self.data_loader.load_scene_data(self.scene_id)
        scene_data = self._scene_data
        
        # Create a copy to avoid modifying original data
        context_data = scene_data.copy()
        
        # Get the target keyframe token for the given keyframe_id (from unsorted list)
        target_keyframe_token = self.data_loader._assign_keyframe_token(self.scene_id, self.keyframe_id)
        
        # Sort keyframes by timestamp first
        sorted_keyframe_tokens = self._sort_keyframes_by_timestamp(scene_data)
        # logger.info(f"Total keyframes: {len(sorted_keyframe_tokens)}")
        
        # keyframe id is 1 to len(sorted_keyframe_tokens)
        if self.keyframe_id > len(sorted_keyframe_tokens):
            logger.error(f"keyframe_id {self.keyframe_id} exceeds available keyframes ({len(sorted_keyframe_tokens)})")
            return None
        
        # Find the position of target keyframe in the sorted list
        if target_keyframe_token not in sorted_keyframe_tokens:
            logger.error(f"Target keyframe token {target_keyframe_token} not found in sorted keyframes")
            return None
        
        target_sorted_index = sorted_keyframe_tokens.index(target_keyframe_token)
        
        # Get keyframes up to and including the target keyframe from the sorted list
        context_keyframe_tokens = sorted_keyframe_tokens[:target_sorted_index+1]
        # logger.info(f"Context keyframe tokens: {context_keyframe_tokens}")
        context_keyframes = {token: scene_data['key_frames'][token] 
                           for token in context_keyframe_tokens}
        # logger.info(f"Context keyframes: {context_keyframes}")
        
        # Get all sample tokens up to and including the target keyframe token
        # Need to determine which samples come before/at the target keyframe
        sample_tokens = list(scene_data['samples'].keys())
        
        # Find the position of target keyframe in samples
        if target_keyframe_token in sample_tokens:
            target_sample_index = sample_tokens.index(target_keyframe_token)
            context_sample_tokens = sample_tokens[:target_sample_index + 1]
        else:
            logger.warning(f"Target keyframe token {target_keyframe_token} not found in samples")
            # Fallback: use all samples up to the last keyframe token we're including
            last_keyframe_token = context_keyframe_tokens[-1]
            if last_keyframe_token in sample_tokens:
                target_sample_index = sample_tokens.index(last_keyframe_token)
                context_sample_tokens = sample_tokens[:target_sample_index + 1]
            else:
                context_sample_tokens = sample_tokens  # fallback to all samples
        
        context_samples = {token: scene_data['samples'][token] 
                         for token in context_sample_tokens}
        
        # Update the context data
        context_data['samples'] = context_samples
        context_data['key_frames'] = context_keyframes
        
        logger.info(f"Context samples count: {len(context_samples)}")
        logger.info(f"Context keyframes count: {len(context_keyframes)}")
        logger.info(f"Target keyframe token: {target_keyframe_token}")
        
        return context_data
    
    def get_qa_pair(self, qa_type, qa_pair_serial):
        # qa_serial is 1 to len(qa_pairs)   

        context_data_for_keyframe = self.get_context_for_keyframe_only()
        qa_pair_by_qa_type = context_data_for_keyframe["key_frames"][self.keyframe_token]["QA"][qa_type]
        # check if qa_pair_serial is valid
        if qa_pair_serial > len(qa_pair_by_qa_type):
            logger.error(f"qa_pair_serial {qa_pair_serial} exceeds available qa pairs ({len(qa_pair_by_qa_type)})")
            return None
        if qa_pair_serial < 1:
            logger.error(f"qa_pair_serial {qa_pair_serial} is less than 1")
            return None
        qa_pair = qa_pair_by_qa_type[qa_pair_serial-1]
        return qa_pair

    def get_annotated_images(self):
        keyframe_data = self.get_context_for_keyframe_only()["key_frames"][self.keyframe_token]
        key_object_infos = keyframe_data.get("key_object_infos", {})
        image_paths = keyframe_data.get("image_paths", {})
        print(f"Found {len(key_object_infos)} objects in frame")
        print(f"Found {len(image_paths)} image paths: {list(image_paths.keys())}")
        for c_tag in key_object_infos:
            print(f"  {c_tag}")
        
        # Create subplots for each camera
        cameras = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        images_processed = 0
        for i, camera in enumerate(cameras):
            if camera in image_paths:
                img_path = image_paths[camera]
                # Adjust path to work with current directory structure
                img_path = img_path.replace("../nuscenes/", "data/v1.0-mini/")
                
                print(f"Processing {camera}: {img_path}")
                img = self.draw_bboxes_on_image(img_path, key_object_infos, camera)
                if img is not None:
                    axes[i].imshow(img)
                    axes[i].set_title(f"{camera}")
                    axes[i].axis('off')
                    images_processed += 1
                else:
                    print(f"Failed to process image for {camera}")
            else:
                print(f"No image path found for {camera}")
        
        print(f"Successfully processed {images_processed} images")
        plt.tight_layout()
            
        # Convert to bytes for model consumption
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        image_bytes = buf.getvalue()
        plt.close()  # Important: close to free memory
        
        print(f"Generated image with {len(image_bytes)} bytes")
        
        # Return as bytes (most flexible for model input)
        return image_bytes

    # Alternative: Return as base64 string if your model expects that format
    def get_annotated_images_base64(self,buf):
        # ... same code as above until buf.seek(0)
        
        # Convert to base64 string
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return image_base64
    
    
    
    def draw_bboxes_on_image(self, image_path, key_object_infos, camera_name):
        """Draw bounding boxes on image for objects in specified camera"""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        
        # Draw bounding boxes for objects in this camera
        for c_tag, obj_info in key_object_infos.items():
            # Parse c-tag to get camera and coordinates
            # Extract camera name from c_tag (format: <id,CAMERA_NAME,x,y>)
            if ',' in c_tag:
                camera_from_tag = c_tag.split(',')[1]
                if camera_name == camera_from_tag:
                    bbox = obj_info["2d_bbox"]  # [x_min, y_min, x_max, y_max]
                    category = obj_info["Category"]
                    # status = obj_info["Status"]  # Unused variable
                    description = obj_info["Visual_description"]
                    
                    print(f"Drawing box for {c_tag}: {category} - {description}")
                    
                    # Clamp bounding box coordinates to image bounds
                    x_min = max(0, int(bbox[0]))
                    y_min = max(0, int(bbox[1]))
                    x_max = min(width, int(bbox[2]))
                    y_max = min(height, int(bbox[3]))
                    
                    # Only draw if the bounding box is valid (has positive area)
                    if x_min < x_max and y_min < y_max:
                        # Draw rectangle
                        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                        
                        # Add label with better visibility and positioning
                        label = f"{category}: {description}"
                        
                        # Calculate text position - try to place above the box, but if that's outside bounds, place below
                        font_scale = 0.6
                        font_thickness = 2
                        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                        
                        # Try to position text above the bounding box
                        text_x = x_min
                        text_y = y_min - 10
                        
                        # If text would go above image, position it below the box
                        if text_y - text_height < 0:
                            text_y = y_max + text_height + 5
                        
                        # If text would go below image, position it inside the box at the top
                        if text_y > height:
                            text_y = y_min + text_height + 5
                        
                        # Ensure text doesn't go outside horizontal bounds
                        if text_x + text_width > width:
                            text_x = width - text_width - 5
                        if text_x < 0:
                            text_x = 5
                        
                        # Draw black outline first, then white text for better visibility
                        cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness + 1)
                        cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
        return img
            
    def get_vehicle_data_upto_sample_token(self):
        """
        Get an overview/aggregate of vehicle data up to the keyframe token.
        
        Returns:
            Dictionary containing aggregated vehicle metrics up to the keyframe
        """
        sample_token = self.keyframe_token
            
        try:
            # Use cached scene data
            if self._scene_data is None:
                self._scene_data = self.data_loader.load_scene_data(self.scene_id)
            scene_data = self._scene_data
            
            samples = scene_data['samples']
            
            # Sort samples by timestamp to ensure chronological order
            sorted_samples = sorted(samples.items(), key=lambda x: x[1]['ego_pose']['timestamp'])
            
            # Find the target sample and get all samples up to it
            target_found = False
            movement_data = []
            timestamps = []
            positions = []
            rotations = []
            
            for sample_token_iter, sample_data in sorted_samples:
                # Stop when we reach the target sample token
                if sample_token_iter == sample_token:
                    target_found = True
                
                ego_pose = sample_data['ego_pose']
                
                # Extract basic pose data
                timestamp = ego_pose['timestamp']
                position = ego_pose['translation']
                rotation = ego_pose['rotation']
                
                # Calculate heading from quaternion
                heading = self._quaternion_to_heading(rotation)
                
                # Store for velocity calculations
                timestamps.append(timestamp)
                positions.append(position)
                rotations.append(rotation)
                
                # Initialize movement data entry
                movement_entry = {
                    'timestamp': timestamp,
                    'position': position,
                    'rotation': rotation,
                    'heading': heading,
                    'velocity': [0, 0, 0],  # Will be calculated
                    'speed': 0.0,
                    'acceleration': [0, 0, 0],  # Will be calculated
                    'angular_velocity': 0.0,
                    'curvature': 0.0,
                    'sample_token': sample_token_iter
                }
                
                movement_data.append(movement_entry)
                
                # Stop processing after reaching the target sample
                if target_found:
                    break
            
            if not target_found:
                logger.warning(f"Sample token {sample_token} not found in scene {self.scene_id}")
                return {}
            
            # Calculate derived metrics using the same methods as DataLoader
            self._calculate_velocity_and_acceleration(movement_data, timestamps)
            self._calculate_curvature(movement_data)
            
            # Calculate summary statistics
            summary_stats = self._calculate_movement_summary(movement_data)
            
            # Return only the overview/aggregate data
            return {
                'scene_id': self.scene_id,
                'target_sample_token': sample_token,
                'nbr_samples': len(movement_data),
                'total_distance': f"{float(summary_stats['total_distance']):.2f} m",
                'total_duration': f"{float(summary_stats['total_duration']):.2f} s",
                'avg_speed': f"{float(summary_stats['avg_speed']):.2f} m/s",
                'max_speed': f"{float(summary_stats['max_speed']):.2f} m/s",
                'avg_acceleration': f"{float(summary_stats['avg_acceleration']):.2f} m/s²",
                'max_acceleration': f"{float(summary_stats['max_acceleration']):.2f} m/s²",
                'avg_curvature': f"{float(summary_stats['avg_curvature']):.2f} rad/m",
                'turning_segments': len(summary_stats['turning_segments']),
                'straight_segments': len(summary_stats['straight_segments']),
                'stopping_periods': len(summary_stats['stopping_periods'])
            }
            
        except Exception as e:
            logger.error(f"Error extracting vehicle data up to sample token {sample_token}: {e}")
            return {}
    
    def _quaternion_to_heading(self, quaternion: List[float]) -> float:
        """
        Convert quaternion rotation to heading angle in radians.
        
        Args:
            quaternion: Quaternion [w, x, y, z]
            
        Returns:
            Heading angle in radians
        """
        w, x, y, z = quaternion
        
        # Calculate heading from quaternion
        heading = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        
        return heading
    
    def _calculate_velocity_and_acceleration(self, movement_data: List[Dict], timestamps: List[int]):
        """
        Calculate velocity and acceleration from position data.
        
        Args:
            movement_data: List of movement data dictionaries
            timestamps: List of timestamps
        """
        for i in range(len(movement_data)):
            if i == 0:
                # First entry - no previous data for velocity/acceleration
                continue
            
            # Calculate time difference in seconds
            dt = (timestamps[i] - timestamps[i-1]) / 1e6  # Convert microseconds to seconds
            
            if dt > 0:
                # Calculate velocity
                curr_pos = np.array(movement_data[i]['position'])
                prev_pos = np.array(movement_data[i-1]['position'])
                velocity = (curr_pos - prev_pos) / dt
                movement_data[i]['velocity'] = velocity.tolist()
                movement_data[i]['speed'] = np.linalg.norm(velocity)
                
                # Calculate angular velocity
                curr_heading = movement_data[i]['heading']
                prev_heading = movement_data[i-1]['heading']
                angular_velocity = (curr_heading - prev_heading) / dt
                movement_data[i]['angular_velocity'] = angular_velocity
                
                if i > 1:
                    # Calculate acceleration
                    curr_vel = np.array(movement_data[i]['velocity'])
                    prev_vel = np.array(movement_data[i-1]['velocity'])
                    acceleration = (curr_vel - prev_vel) / dt
                    movement_data[i]['acceleration'] = acceleration.tolist()
    
    def _calculate_curvature(self, movement_data: List[Dict]):
        """
        Calculate path curvature from heading changes.
        
        Args:
            movement_data: List of movement data dictionaries
        """
        for i in range(1, len(movement_data)):
            if i == 0:
                continue
            
            # Calculate heading change
            heading_change = abs(movement_data[i]['heading'] - movement_data[i-1]['heading'])
            
            # Calculate distance traveled
            curr_pos = np.array(movement_data[i]['position'])
            prev_pos = np.array(movement_data[i-1]['position'])
            distance = np.linalg.norm(curr_pos - prev_pos)
            
            # Calculate curvature (approximate)
            if distance > 0:
                curvature = heading_change / distance
                movement_data[i]['curvature'] = curvature
            else:
                movement_data[i]['curvature'] = 0.0
    
    def _calculate_movement_summary(self, movement_data: List[Dict]) -> Dict[str, Any]:
        """
        Calculate summary statistics for movement data.
        
        Args:
            movement_data: List of movement data dictionaries
            
        Returns:
            Dictionary containing summary statistics
        """
        if not movement_data:
            return {}
        
        # Extract metrics
        speeds = [entry['speed'] for entry in movement_data if entry['speed'] > 0]
        accelerations = [np.linalg.norm(entry['acceleration']) for entry in movement_data if any(entry['acceleration'])]
        curvatures = [entry['curvature'] for entry in movement_data if entry['curvature'] > 0]
        
        # Calculate total distance
        total_distance = 0.0
        for i in range(1, len(movement_data)):
            curr_pos = np.array(movement_data[i]['position'])
            prev_pos = np.array(movement_data[i-1]['position'])
            total_distance += np.linalg.norm(curr_pos - prev_pos)
        
        # Calculate total duration
        if len(movement_data) > 1:
            total_duration = (movement_data[-1]['timestamp'] - movement_data[0]['timestamp']) / 1e6  # seconds
        else:
            total_duration = 0.0
        
        # Identify movement segments
        turning_segments = []
        straight_segments = []
        stopping_periods = []
        
        # Simple segmentation based on curvature and speed
        for i, entry in enumerate(movement_data):
            if entry['curvature'] > 0.01:  # High curvature = turning
                turning_segments.append(i)
            elif entry['speed'] < 0.5:  # Low speed = stopping
                stopping_periods.append(i)
            else:  # Medium curvature and speed = straight
                straight_segments.append(i)
        
        return {
            'total_distance': total_distance,
            'total_duration': total_duration,
            'avg_speed': np.mean(speeds) if speeds else 0.0,
            'max_speed': np.max(speeds) if speeds else 0.0,
            'avg_acceleration': np.mean(accelerations) if accelerations else 0.0,
            'max_acceleration': np.max(accelerations) if accelerations else 0.0,
            'avg_curvature': np.mean(curvatures) if curvatures else 0.0,
            'turning_segments': turning_segments,
            'straight_segments': straight_segments,
            'stopping_periods': stopping_periods
        }

    def get_sensor_data_upto_sample_token(self):
        """
        Get sensor data summary for the specific keyframe token.
        
        Returns:
            Dictionary containing sensor detection data and object annotations
        """
        sample_token = self.keyframe_token
        
        try:
            # Use cached scene data
            if self._scene_data is None:
                self._scene_data = self.data_loader.load_scene_data(self.scene_id)
            scene_data = self._scene_data
            
            # Get the specific sample for this keyframe
            if sample_token not in scene_data['samples']:
                logger.error(f"Sample token {sample_token} not found in scene {self.scene_id}")
                return {}
            
            sample_data = scene_data['samples'][sample_token]
            annotations = sample_data['annotations']
            sensor_data = sample_data['sensor_data']
            
            # Process annotations to get object detection summary
            object_summary = {}
            category_counts = {}
            sensor_detection_stats = {'lidar_detections': 0, 'radar_detections': 0}
            visibility_stats = {}
            
            for annotation in annotations:
                category = annotation['category']
                category_counts[category] = category_counts.get(category, 0) + 1
                
                # Count sensor detections
                sensor_detection_stats['lidar_detections'] += annotation['num_lidar_pts']
                sensor_detection_stats['radar_detections'] += annotation['num_radar_pts']
                
                # Track visibility levels
                visibility_level = annotation['visibility']['level']
                visibility_stats[visibility_level] = visibility_stats.get(visibility_level, 0) + 1
            
            # Create sensor data summary
            sensor_summary = {
                'scene_id': self.scene_id,
                'sample_token': sample_token,
                'timestamp': sample_data['timestamp'],
                
                # Object detection summary
                'object_detection': {
                    'total_objects': len(annotations),
                    'category_counts': category_counts,
                    'unique_categories': list(category_counts.keys()),
                    'num_categories': len(category_counts)
                },
                
                # Sensor detection statistics
                'sensor_detections': {
                    'total_lidar_points': sensor_detection_stats['lidar_detections'],
                    'total_radar_points': sensor_detection_stats['radar_detections'],
                    'objects_with_lidar': len([a for a in annotations if a['num_lidar_pts'] > 0]),
                    'objects_with_radar': len([a for a in annotations if a['num_radar_pts'] > 0])
                },
                
                # Visibility statistics
                'visibility_distribution': visibility_stats,
                
                # Available sensors
                'available_sensors': {
                    'sensor_types': list(sensor_data.keys()),
                    'camera_sensors': [s for s in sensor_data.keys() if 'CAM_' in s],
                    'radar_sensors': [s for s in sensor_data.keys() if 'RADAR_' in s],
                    'lidar_sensors': [s for s in sensor_data.keys() if 'LIDAR_' in s]
                }
            }
            
            logger.info(f"Retrieved sensor data for sample token: {sample_token} with {len(annotations)} detected objects")
            return sensor_summary
            
        except Exception as e:
            logger.error(f"Error extracting sensor data for sample token {sample_token}: {e}")
            return {}