#!/usr/bin/env python3
"""
Find Overlapping Scenes Script

This script identifies overlapping scenes between NuScenes-mini and DriveLM datasets.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Set
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_nuscenes_scenes(nuscenes_scene_file: str) -> Dict[str, Dict]:
    """
    Load NuScenes scene data.
    
    Args:
        nuscenes_scene_file: Path to NuScenes scene.json file
        
    Returns:
        Dictionary mapping scene tokens to scene data
    """
    logger.info(f"Loading NuScenes scenes from {nuscenes_scene_file}")
    
    with open(nuscenes_scene_file, 'r') as f:
        scenes_data = json.load(f)
    
    scenes_dict = {}
    for scene in scenes_data:
        scenes_dict[scene['token']] = {
            'name': scene['name'],
            'description': scene['description'],
            'nbr_samples': scene['nbr_samples']
        }
    
    logger.info(f"Loaded {len(scenes_dict)} NuScenes scenes")
    return scenes_dict

def load_drivelm_scenes(drivelm_file: str) -> Dict[str, Dict]:
    """
    Load DriveLM scene data.
    
    Args:
        drivelm_file: Path to DriveLM training JSON file
        
    Returns:
        Dictionary mapping scene tokens to scene data
    """
    logger.info(f"Loading DriveLM scenes from {drivelm_file}")
    
    with open(drivelm_file, 'r') as f:
        drivelm_data = json.load(f)
    
    scenes_dict = {}
    for scene_token, scene_data in drivelm_data.items():
        scenes_dict[scene_token] = {
            'description': scene_data.get('scene_description', ''),
            'key_frames_count': len(scene_data.get('key_frames', {}))
        }
    
    logger.info(f"Loaded {len(scenes_dict)} DriveLM scenes")
    return scenes_dict

def find_overlapping_scenes(nuscenes_scenes: Dict[str, Dict], 
                          drivelm_scenes: Dict[str, Dict]) -> List[Dict]:
    """
    Find overlapping scenes between NuScenes and DriveLM datasets.
    
    Args:
        nuscenes_scenes: NuScenes scene data
        drivelm_scenes: DriveLM scene data
        
    Returns:
        List of overlapping scene information
    """
    logger.info("Finding overlapping scenes...")
    
    nuscenes_tokens = set(nuscenes_scenes.keys())
    drivelm_tokens = set(drivelm_scenes.keys())
    
    overlapping_tokens = nuscenes_tokens.intersection(drivelm_tokens)
    
    logger.info(f"Found {len(overlapping_tokens)} overlapping scenes")
    
    overlapping_scenes = []
    for token in overlapping_tokens:
        nuscenes_info = nuscenes_scenes[token]
        drivelm_info = drivelm_scenes[token]
        
        overlapping_scenes.append({
            'scene_token': token,
            'scene_name': nuscenes_info['name'],
            'nuscenes_description': nuscenes_info['description'],
            'nuscenes_samples': nuscenes_info['nbr_samples'],
            'drivelm_description': drivelm_info['description'],
            'drivelm_key_frames': drivelm_info['key_frames_count']
        })
    
    return overlapping_scenes

def save_overlapping_scenes(overlapping_scenes: List[Dict], output_file: str):
    """
    Save overlapping scenes to CSV file.
    
    Args:
        overlapping_scenes: List of overlapping scene data
        output_file: Output CSV file path
    """
    logger.info(f"Saving overlapping scenes to {output_file}")
    
    df = pd.DataFrame(overlapping_scenes)
    df.to_csv(output_file, index=False)
    
    logger.info(f"Saved {len(overlapping_scenes)} overlapping scenes to {output_file}")

def main():
    """Main function to find overlapping scenes."""
    
    # Define file paths
    nuscenes_scene_file = "data/v1.0-mini/v1.0-mini/scene.json"
    drivelm_file = "data/v1_1_train_nus.json"
    output_file = "data/concatenated_data/overlapping_scenes_analysis.csv"
    
    # Check if files exist
    if not Path(nuscenes_scene_file).exists():
        logger.error(f"NuScenes scene file not found: {nuscenes_scene_file}")
        return
    
    if not Path(drivelm_file).exists():
        logger.error(f"DriveLM file not found: {drivelm_file}")
        return
    
    # Load scene data
    nuscenes_scenes = load_nuscenes_scenes(nuscenes_scene_file)
    drivelm_scenes = load_drivelm_scenes(drivelm_file)
    
    # Find overlapping scenes
    overlapping_scenes = find_overlapping_scenes(nuscenes_scenes, drivelm_scenes)
    
    # Save results
    save_overlapping_scenes(overlapping_scenes, output_file)
    
    # Print summary
    logger.info("=== OVERLAPPING SCENES SUMMARY ===")
    logger.info(f"NuScenes-mini total scenes: {len(nuscenes_scenes)}")
    logger.info(f"DriveLM total scenes: {len(drivelm_scenes)}")
    logger.info(f"Overlapping scenes: {len(overlapping_scenes)}")
    logger.info(f"Overlap percentage: {len(overlapping_scenes)/len(nuscenes_scenes)*100:.1f}%")
    
    # Print overlapping scene details
    for scene in overlapping_scenes:
        logger.info(f"Scene: {scene['scene_name']} ({scene['scene_token'][:8]}...)")
        logger.info(f"  NuScenes: {scene['nuscenes_samples']} samples")
        logger.info(f"  DriveLM: {scene['drivelm_key_frames']} keyframes")
        logger.info(f"  NuScenes desc: {scene['nuscenes_description']}")
        logger.info(f"  DriveLM desc: {scene['drivelm_description']}")
        logger.info("")

if __name__ == "__main__":
    main() 