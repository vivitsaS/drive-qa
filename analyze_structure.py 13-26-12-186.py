#!/usr/bin/env python3

import json
from analysis.data_loader import DataLoader

def analyze_first_entry():
    """Analyze the structure of the first entry using DataLoader"""
    
    # Initialize data loader
    loader = DataLoader("data/concatenated_data/first_entry.json")
    
    # Load all data
    all_data = loader.load_all_data()
    print(f"Number of scenes in first entry: {len(all_data)}")
    
    # Get the first scene token
    scene_token = list(all_data.keys())[0]
    print(f"\nFirst scene token: {scene_token}")
    
    # Load scene data
    scene_data = loader.load_scene_data(scene_token)
    
    print(f"\nScene name: {scene_data.get('scene_name')}")
    print(f"Scene description: {scene_data.get('scene_description')}")
    print(f"Number of samples: {scene_data.get('nbr_samples')}")
    print(f"Log token: {scene_data.get('log_token')}")
    
    # Analyze samples
    samples = scene_data.get('samples', {})
    print(f"\nNumber of samples: {len(samples)}")
    if samples:
        first_sample_token = list(samples.keys())[0]
        first_sample = samples[first_sample_token]
        print(f"First sample token: {first_sample_token}")
        print(f"First sample keys: {list(first_sample.keys())}")
        
        # Analyze sensor data
        sensor_data = first_sample.get('sensor_data', {})
        print(f"\nSensor types: {list(sensor_data.keys())}")
        
        for sensor_type, sensor_info in sensor_data.items():
            print(f"\n{sensor_type}:")
            print(f"  - Token: {sensor_info.get('token')}")
            print(f"  - File format: {sensor_info.get('fileformat')}")
            print(f"  - Is key frame: {sensor_info.get('is_key_frame')}")
            print(f"  - Filename: {sensor_info.get('filename', 'N/A')}")
    
    # Analyze key frames
    key_frames = scene_data.get('key_frames', {})
    print(f"\nNumber of key frames: {len(key_frames)}")
    
    if key_frames:
        first_keyframe_token = list(key_frames.keys())[0]
        first_keyframe = key_frames[first_keyframe_token]
        print(f"\nFirst keyframe token: {first_keyframe_token}")
        print(f"First keyframe keys: {list(first_keyframe.keys())}")
        
        # Analyze QA data
        qa_pairs = first_keyframe.get('QA', {})
        print(f"\nQA pairs structure: {type(qa_pairs)}")
        if isinstance(qa_pairs, dict):
            print(f"Number of QA pairs in first keyframe: {len(qa_pairs)}")
            if qa_pairs:
                first_qa_key = list(qa_pairs.keys())[0]
                first_qa = qa_pairs[first_qa_key]
                print(f"\nFirst QA pair key: {first_qa_key}")
                print(f"First QA pair structure:")
                print(f"  Keys: {list(first_qa.keys())}")
                for key, value in first_qa.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"  {key}: {value[:100]}...")
                    else:
                        print(f"  {key}: {value}")
        elif isinstance(qa_pairs, list):
            print(f"Number of QA pairs in first keyframe: {len(qa_pairs)}")
            if qa_pairs:
                first_qa = qa_pairs[0]
                print(f"\nFirst QA pair structure:")
                print(f"  Keys: {list(first_qa.keys())}")
                for key, value in first_qa.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"  {key}: {value[:100]}...")
                    else:
                        print(f"  {key}: {value}")

if __name__ == "__main__":
    analyze_first_entry() 