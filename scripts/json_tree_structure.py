#!/usr/bin/env python3
"""
Script to generate a concise tree structure of the first_entry.json file.
Shows the nested hierarchy without repetitive details.
"""

import json
import sys
from pathlib import Path


def print_tree_structure(data, prefix="", max_depth=4, current_depth=0):
    """
    Recursively print the tree structure of JSON data.
    
    Args:
        data: The data to print
        prefix: Prefix for indentation
        max_depth: Maximum depth to traverse
        current_depth: Current depth level
    """
    if current_depth >= max_depth:
        print(f"{prefix}... (max depth reached)")
        return
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                if isinstance(value, list) and len(value) > 0:
                    print(f"{prefix}├── {key}: [{len(value)} items]")
                    # Show only first item structure for lists
                    if len(value) > 0:
                        print_tree_structure(value[0], prefix + "│   ", max_depth, current_depth + 1)
                        if len(value) > 1:
                            print(f"{prefix}│   └── ... ({len(value) - 1} more similar items)")
                else:
                    print(f"{prefix}├── {key}: {type(value).__name__}")
                    print_tree_structure(value, prefix + "│   ", max_depth, current_depth + 1)
            else:
                # Truncate long values for readability
                value_str = str(value)
                if len(value_str) > 30:
                    value_str = value_str[:27] + "..."
                print(f"{prefix}├── {key}: {value_str}")
    elif isinstance(data, list):
        if len(data) > 0:
            print(f"{prefix}├── [{len(data)} items]")
            # Show only first item structure
            if len(data) > 0:
                print_tree_structure(data[0], prefix + "│   ", max_depth, current_depth + 1)
                if len(data) > 1:
                    print(f"{prefix}└── ... ({len(data) - 1} more similar items)")


def main():
    """Main function to load and display the tree structure."""
    json_file = Path("concatenated_data/first_entry.json")
    
    if not json_file.exists():
        print(f"Error: {json_file} not found!")
        sys.exit(1)
    
    try:
        print(f"Loading {json_file}...")
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        print("\n" + "="*60)
        print("CONCISE TREE STRUCTURE OF first_entry.json")
        print("="*60)
        
        # Get the first (and only) scene
        scene_token = list(data.keys())[0]
        scene_data = data[scene_token]
        
        print(f"Scene Token: {scene_token}")
        print("Structure:")
        print_tree_structure(scene_data, max_depth=5)
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        # Print some statistics
        if 'samples' in scene_data:
            num_samples = len(scene_data['samples'])
            print(f"Number of samples: {num_samples}")
            
            # Count sensor types in first sample
            if scene_data['samples']:
                first_sample = list(scene_data['samples'].values())[0]
                if 'sensor_data' in first_sample:
                    sensor_types = list(first_sample['sensor_data'].keys())
                    print(f"Sensor types: {', '.join(sensor_types)}")
                
                if 'QA' in first_sample:
                    qa_categories = list(first_sample['QA'].keys())
                    print(f"QA categories: {', '.join(qa_categories)}")
                    
                    # Count QA pairs
                    total_qa_pairs = 0
                    for category in qa_categories:
                        total_qa_pairs += len(first_sample['QA'][category])
                    print(f"QA pairs in first sample: {total_qa_pairs}")
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 