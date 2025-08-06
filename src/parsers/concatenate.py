#!/usr/bin/env python3
"""
NuScenes + Training JSON Concatenation Script

This script merges NuScenes data with training JSON data for the 6 scenes
to create a unified data structure with all necessary fields.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NuScenesConcatenator:
    """Concatenates NuScenes data with training JSON data."""
    
    def __init__(self, nuscenes_dir: str, training_json_path: str):
        """
        Initialize the concatenator.
        
        Args:
            nuscenes_dir: Path to NuScenes data directory
            training_json_path: Path to training JSON file
        """
        self.nuscenes_dir = Path(nuscenes_dir)
        self.training_json_path = Path(training_json_path)
        self.output_path = Path("concatenated_data.json")
        
        # Data storage
        self.nuscenes_data = {}
        self.training_data = {}
        self.concatenated_data = {}
        
        # Lookup dictionaries for efficient access
        self.samples_by_scene = {}
        self.annotations_by_sample = {}
        self.sensor_data_by_sample = {}
        self.poses_by_sample = {}
        self.categories = {}
        self.attributes = {}
        self.visibility = {}
        self.instances = {}
        self.calibrated_sensors = {}
        
    def load_nuscenes_data(self):
        """Load all NuScenes metadata files."""
        logger.info("Loading NuScenes data...")
        
        # Load core metadata files
        metadata_files = [
            "scene.json", "sample.json", "sample_annotation.json",
            "sample_data.json", "ego_pose.json", "category.json",
            "attribute.json", "visibility.json", "instance.json",
            "calibrated_sensor.json", "sensor.json"
        ]
        
        for filename in metadata_files:
            file_path = self.nuscenes_dir / "v1.0-mini" / filename
            if file_path.exists():
                with open(file_path, 'r') as f:
                    self.nuscenes_data[filename.replace('.json', '')] = json.load(f)
                logger.info(f"Loaded {filename}")
            else:
                logger.warning(f"File not found: {file_path}")
        
        # Create lookup dictionaries
        self._create_lookup_dictionaries()
        
    def _create_lookup_dictionaries(self):
        """Create efficient lookup dictionaries for data access."""
        logger.info("Creating lookup dictionaries...")
        
        # Samples by scene
        for sample in self.nuscenes_data.get('sample', []):
            scene_token = sample['scene_token']
            if scene_token not in self.samples_by_scene:
                self.samples_by_scene[scene_token] = []
            self.samples_by_scene[scene_token].append(sample)
        
        # Annotations by sample
        for annotation in self.nuscenes_data.get('sample_annotation', []):
            sample_token = annotation['sample_token']
            if sample_token not in self.annotations_by_sample:
                self.annotations_by_sample[sample_token] = []
            self.annotations_by_sample[sample_token].append(annotation)
        
        # Sensor data by sample
        for sensor_data in self.nuscenes_data.get('sample_data', []):
            sample_token = sensor_data['sample_token']
            if sample_token not in self.sensor_data_by_sample:
                self.sensor_data_by_sample[sample_token] = []
            self.sensor_data_by_sample[sample_token].append(sensor_data)
        
        # Poses by sample
        for pose in self.nuscenes_data.get('ego_pose', []):
            self.poses_by_sample[pose['token']] = pose
        
        # Categories, attributes, visibility, instances
        for category in self.nuscenes_data.get('category', []):
            self.categories[category['token']] = category
        
        for attribute in self.nuscenes_data.get('attribute', []):
            self.attributes[attribute['token']] = attribute
        
        for vis in self.nuscenes_data.get('visibility', []):
            self.visibility[vis['token']] = vis
        
        for instance in self.nuscenes_data.get('instance', []):
            self.instances[instance['token']] = instance
        
        for sensor in self.nuscenes_data.get('calibrated_sensor', []):
            self.calibrated_sensors[sensor['token']] = sensor
        
        logger.info("Lookup dictionaries created")
        
    def load_training_data(self):
        """Load training JSON data."""
        logger.info("Loading training JSON data...")
        
        with open(self.training_json_path, 'r') as f:
            self.training_data = json.load(f)
        
        logger.info(f"Loaded {len(self.training_data)} scenes from training JSON")
        
    def validate_scenes(self):
        """Validate that all training scenes exist in NuScenes data."""
        logger.info("Validating scenes...")
        
        training_scenes = set(self.training_data.keys())
        nuscenes_scenes = set()
        
        for scene in self.nuscenes_data.get('scene', []):
            nuscenes_scenes.add(scene['token'])
        
        missing_scenes = training_scenes - nuscenes_scenes
        if missing_scenes:
            logger.error(f"Missing scenes in NuScenes data: {missing_scenes}")
            return False
        
        logger.info(f"All {len(training_scenes)} training scenes found in NuScenes data")
        return True
        
    def get_enriched_annotation(self, annotation: Dict) -> Dict:
        """Enrich annotation with category, attribute, and visibility information."""
        enriched = annotation.copy()
        
        # Add category information
        if 'instance_token' in annotation:
            instance = self.instances.get(annotation['instance_token'])
            if instance:
                category = self.categories.get(instance['category_token'])
                if category:
                    enriched['category'] = category['name']
                    enriched['category_description'] = category['description']
        
        # Add attribute information
        if 'attribute_tokens' in annotation:
            enriched['attributes'] = []
            for attr_token in annotation['attribute_tokens']:
                attr = self.attributes.get(attr_token)
                if attr:
                    enriched['attributes'].append({
                        'name': attr['name'],
                        'description': attr['description']
                    })
        
        # Add visibility information
        if 'visibility_token' in annotation:
            vis = self.visibility.get(annotation['visibility_token'])
            if vis:
                enriched['visibility'] = {
                    'level': vis['level'],
                    'description': vis['description']
                }
        
        return enriched
        
    def get_sensor_data_for_sample(self, sample_token: str) -> Dict:
        """Get all sensor data for a sample."""
        sensor_data = {}
        
        for data in self.sensor_data_by_sample.get(sample_token, []):
            # Extract sensor channel from filename
            filename = data.get('filename', '')
            if 'CAM_FRONT' in filename:
                channel = 'CAM_FRONT'
            elif 'CAM_BACK' in filename:
                channel = 'CAM_BACK'
            elif 'CAM_FRONT_LEFT' in filename:
                channel = 'CAM_FRONT_LEFT'
            elif 'CAM_FRONT_RIGHT' in filename:
                channel = 'CAM_FRONT_RIGHT'
            elif 'CAM_BACK_LEFT' in filename:
                channel = 'CAM_BACK_LEFT'
            elif 'CAM_BACK_RIGHT' in filename:
                channel = 'CAM_BACK_RIGHT'
            elif 'LIDAR_TOP' in filename:
                channel = 'LIDAR_TOP'
            elif 'RADAR' in filename:
                # Extract specific radar channel
                if 'RADAR_FRONT' in filename:
                    channel = 'RADAR_FRONT'
                elif 'RADAR_FRONT_LEFT' in filename:
                    channel = 'RADAR_FRONT_LEFT'
                elif 'RADAR_FRONT_RIGHT' in filename:
                    channel = 'RADAR_FRONT_RIGHT'
                elif 'RADAR_BACK_LEFT' in filename:
                    channel = 'RADAR_BACK_LEFT'
                elif 'RADAR_BACK_RIGHT' in filename:
                    channel = 'RADAR_BACK_RIGHT'
                else:
                    channel = 'RADAR_UNKNOWN'
            else:
                channel = 'UNKNOWN'
            
            # Add calibration data
            calibrated_sensor = self.calibrated_sensors.get(data.get('calibrated_sensor_token', ''))
            if calibrated_sensor:
                data['calibrated_sensor'] = calibrated_sensor
            
            sensor_data[channel] = data
        
        return sensor_data
        
    def create_concatenated_scene(self, scene_token: str) -> Dict:
        """Create concatenated data structure for a single scene."""
        logger.info(f"Processing scene {scene_token}")
        
        # Get scene information
        scene_info = None
        for scene in self.nuscenes_data.get('scene', []):
            if scene['token'] == scene_token:
                scene_info = scene
                break
        
        if not scene_info:
            logger.error(f"Scene {scene_token} not found in NuScenes data")
            return {}
        
        # Get training data for this scene
        training_scene = self.training_data.get(scene_token, {})
        
        # Create scene structure
        concatenated_scene = {
            'scene_token': scene_token,
            'scene_name': scene_info.get('name', ''),
            'scene_description': training_scene.get('scene_description', ''),
            'log_token': scene_info.get('log_token', ''),
            'nbr_samples': scene_info.get('nbr_samples', 0),
            'first_sample_token': scene_info.get('first_sample_token', ''),
            'last_sample_token': scene_info.get('last_sample_token', ''),
            'samples': {},
            'key_frames': training_scene.get('key_frames', {})
        }
        
        # Process all samples for this scene
        samples = self.samples_by_scene.get(scene_token, [])
        for sample in samples:
            sample_token = sample['token']
            
            # Get sensor data
            sensor_data = self.get_sensor_data_for_sample(sample_token)
            
            # Get ego pose
            ego_pose = None
            for sensor_data_item in sensor_data.values():
                pose_token = sensor_data_item.get('ego_pose_token')
                if pose_token:
                    ego_pose = self.poses_by_sample.get(pose_token)
                    break
            
            # Get annotations
            annotations = []
            for annotation in self.annotations_by_sample.get(sample_token, []):
                enriched_annotation = self.get_enriched_annotation(annotation)
                annotations.append(enriched_annotation)
            
            # Create sample structure
            sample_data = {
                'token': sample_token,
                'timestamp': sample.get('timestamp'),
                'prev': sample.get('prev'),
                'next': sample.get('next'),
                'sensor_data': sensor_data,
                'ego_pose': ego_pose,
                'annotations': annotations
            }
            
            concatenated_scene['samples'][sample_token] = sample_data
        
        return concatenated_scene
        
    def concatenate_all_scenes(self):
        """Concatenate data for all training scenes."""
        logger.info("Starting concatenation process...")
        
        if not self.validate_scenes():
            logger.error("Scene validation failed")
            return
        
        self.concatenated_data = {}
        
        for scene_token in self.training_data.keys():
            concatenated_scene = self.create_concatenated_scene(scene_token)
            if concatenated_scene:
                self.concatenated_data[scene_token] = concatenated_scene
        
        logger.info(f"Concatenated {len(self.concatenated_data)} scenes")
        
    def save_concatenated_data(self):
        """Save concatenated data to file."""
        logger.info(f"Saving concatenated data to {self.output_path}")
        
        with open(self.output_path, 'w') as f:
            json.dump(self.concatenated_data, f, indent=2)
        
        logger.info("Data saved successfully")
        
    def print_statistics(self):
        """Print statistics about the concatenated data."""
        logger.info("=== CONCATENATED DATA STATISTICS ===")
        
        total_samples = 0
        total_annotations = 0
        total_sensor_data = 0
        
        for scene_token, scene_data in self.concatenated_data.items():
            samples_count = len(scene_data['samples'])
            total_samples += samples_count
            
            scene_annotations = 0
            scene_sensor_data = 0
            
            for sample in scene_data['samples'].values():
                scene_annotations += len(sample['annotations'])
                scene_sensor_data += len(sample['sensor_data'])
            
            total_annotations += scene_annotations
            total_sensor_data += scene_sensor_data
            
            logger.info(f"Scene {scene_data['scene_name']}: {samples_count} samples, "
                       f"{scene_annotations} annotations, {scene_sensor_data} sensor data entries")
        
        logger.info(f"Total: {len(self.concatenated_data)} scenes, {total_samples} samples, "
                   f"{total_annotations} annotations, {total_sensor_data} sensor data entries")
        
    def run(self):
        """Run the complete concatenation process."""
        logger.info("Starting NuScenes + Training JSON concatenation...")
        
        # Load data
        self.load_nuscenes_data()
        self.load_training_data()
        
        # Concatenate
        self.concatenate_all_scenes()
        
        # Save and report
        self.save_concatenated_data()
        self.print_statistics()
        
        logger.info("Concatenation complete!")

def main():
    """Main function."""
    # Define paths
    nuscenes_dir = "data/v1.0-mini"
    training_json_path = "data/v1.0-mini/v1_1_train_overlapping_scenes.json"
    
    # Check if files exist
    if not os.path.exists(nuscenes_dir):
        logger.error(f"NuScenes directory not found: {nuscenes_dir}")
        return
    
    if not os.path.exists(training_json_path):
        logger.error(f"Training JSON not found: {training_json_path}")
        return
    
    # Run concatenation
    concatenator = NuScenesConcatenator(nuscenes_dir, training_json_path)
    concatenator.run()

if __name__ == "__main__":
    main() 