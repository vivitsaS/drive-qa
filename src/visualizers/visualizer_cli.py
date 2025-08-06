#!/usr/bin/env python3

import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_scene_data(json_file, scene_token):
    """Load specific scene data from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data.get(scene_token)

def draw_bboxes_on_image(image_path, key_object_infos, camera_name):
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
                status = obj_info["Status"]
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

def visualize_scene(json_file, scene_token, frame_token=None):
    """Visualize a scene with bounding boxes"""
    scene_data = load_scene_data(json_file, scene_token)
    if not scene_data:
        print(f"Scene {scene_token} not found")
        return
    
    # Get key frames
    key_frames = scene_data.get("key_frames", {})
    if not key_frames:
        print("No key frames found")
        return
    
    # Use first frame if none specified
    if frame_token is None:
        frame_token = list(key_frames.keys())[0]
    
    frame_data = key_frames.get(frame_token)
    if not frame_data:
        print(f"Frame {frame_token} not found")
        return
    
    key_object_infos = frame_data.get("key_object_infos", {})
    image_paths = frame_data.get("image_paths", {})
    
    print(f"Found {len(key_object_infos)} objects in frame")
    for c_tag in key_object_infos:
        print(f"  {c_tag}")
    
    # Create subplots for each camera
    cameras = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, camera in enumerate(cameras):
        if camera in image_paths:
            img_path = image_paths[camera]
            # Adjust path to work with current directory structure
            img_path = img_path.replace("../nuscenes/", "data/v1.0-mini/")
            
            print(f"Processing {camera}: {img_path}")
            img = draw_bboxes_on_image(img_path, key_object_infos, camera)
            if img is not None:
                axes[i].imshow(img)
                axes[i].set_title(f"{camera}")
                axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def list_scenes(json_file):
    """List all available scenes"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    scenes = list(data.keys())
    print(f"Found {len(scenes)} scenes:")
    for i, scene_token in enumerate(scenes):
        print(f"  {i+1}. {scene_token}")
    return scenes

def list_keyframes(scene_data):
    """List all keyframes for a scene with object counts"""
    key_frames = scene_data.get("key_frames", {})
    if not key_frames:
        print("No key frames found")
        return []
    
    print(f"\nFound {len(key_frames)} key frames:")
    frame_tokens = list(key_frames.keys())
    
    for i, frame_token in enumerate(frame_tokens):
        frame_data = key_frames[frame_token]
        key_object_infos = frame_data.get("key_object_infos", {})
        
        # Group objects by camera
        cameras = {}
        for c_tag in key_object_infos:
            if ',' in c_tag:
                camera = c_tag.split(',')[1]
                if camera not in cameras:
                    cameras[camera] = []
                cameras[camera].append(c_tag)
        
        print(f"  {i+1}. Frame {frame_token[:8]}... ({len(key_object_infos)} objects)")
        for camera, objects in cameras.items():
            print(f"      {camera}: {len(objects)} objects")
    
    return frame_tokens

def main():
    """Main CLI interface"""
    json_file = "concatenated_data.json"
    
    print("=== NuScenes Visualizer CLI ===")
    print()
    
    # Check if file exists
    if not Path(json_file).exists():
        print(f"Error: {json_file} not found!")
        return
    
    # List available scenes
    scenes = list_scenes(json_file)
    if not scenes:
        print("No scenes found in data")
        return
    
    # Ask user to select scene
    while True:
        try:
            choice = input(f"\nSelect scene (1-{len(scenes)}): ")
            scene_idx = int(choice) - 1
            if 0 <= scene_idx < len(scenes):
                break
            else:
                print(f"Please enter a number between 1 and {len(scenes)}")
        except ValueError:
            print("Please enter a valid number")
    
    selected_scene = scenes[scene_idx]
    print(f"\nSelected scene: {selected_scene}")
    
    # Load scene data
    scene_data = load_scene_data(json_file, selected_scene)
    if not scene_data:
        print(f"Scene {selected_scene} not found")
        return
    
    # List keyframes for this scene
    frame_tokens = list_keyframes(scene_data)
    if not frame_tokens:
        return
    
    # Ask user to select keyframe
    while True:
        try:
            choice = input(f"\nSelect keyframe (1-{len(frame_tokens)}): ")
            frame_idx = int(choice) - 1
            if 0 <= frame_idx < len(frame_tokens):
                break
            else:
                print(f"Please enter a number between 1 and {len(frame_tokens)}")
        except ValueError:
            print("Please enter a valid number")
    
    selected_frame = frame_tokens[frame_idx]
    print(f"\nSelected frame: {selected_frame}")
    
    # Visualize the selected scene and frame
    print("\nGenerating visualization...")
    visualize_scene(json_file, selected_scene, selected_frame)

if __name__ == "__main__":
    main() 