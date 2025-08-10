#!/usr/bin/env python3

"""
Test script for ImageAnalyzer
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analysis.data_loader import DataLoader
from analysis.image_analysis import ImageAnalyzer
from loguru import logger

def test_image_analyzer():
    """Test the ImageAnalyzer with scene data"""
    
    print("🚗 Testing ImageAnalyzer with nuScenes Sensor Suite")
    print("=" * 60)
    
    try:
        # Initialize data loader and image analyzer
        data_loader = DataLoader()
        image_analyzer = ImageAnalyzer(data_loader)
        
        # Test with scene 1
        scene_id = 1
        print(f"\n📊 Analyzing Scene {scene_id}")
        print("-" * 40)
        
        # Test visual insights
        print("\n🔍 Visual Insights:")
        visual_insights = image_analyzer.analyze_visual_insights(scene_id)
        if visual_insights:
            print(f"  • Total objects detected: {visual_insights.get('object_detection_insights', {}).get('total_objects_detected', 0)}")
            print(f"  • Unique object categories: {visual_insights.get('object_detection_insights', {}).get('unique_object_categories', 0)}")
            print(f"  • Traffic density: {visual_insights.get('scene_understanding', {}).get('traffic_density', 'unknown')}")
            print(f"  • Environment complexity: {visual_insights.get('scene_understanding', {}).get('environment_complexity', 'unknown')}")
        else:
            print("  ❌ No visual insights available")
        
        # Test comprehensive sensor fusion
        print("\n🔗 Comprehensive Sensor Fusion:")
        fusion_insights = image_analyzer.analyze_comprehensive_sensor_fusion(scene_id)
        if fusion_insights:
            availability = fusion_insights.get('sensor_availability', {})
            print(f"  • Total samples: {availability.get('total_samples', 0)}")
            print(f"  • Samples with full suite: {availability.get('samples_with_full_suite', 0)}")
            print(f"  • Full suite coverage rate: {availability.get('full_suite_coverage_rate', 0):.2%}")
            print(f"  • Average cameras per sample: {availability.get('average_cameras_per_sample', 0):.1f}")
            print(f"  • Average radar per sample: {availability.get('average_radar_per_sample', 0):.1f}")
            print(f"  • LiDAR coverage rate: {availability.get('lidar_coverage_rate', 0):.2%}")
            
            coverage = fusion_insights.get('360_degree_coverage', {})
            print(f"  • 360° visual coverage: {coverage.get('visual_coverage', False)}")
            print(f"  • 360° radar coverage: {coverage.get('radar_coverage', False)}")
            print(f"  • LiDAR coverage: {coverage.get('lidar_coverage', False)}")
            print(f"  • Comprehensive coverage: {coverage.get('comprehensive_coverage', False)}")
        else:
            print("  ❌ No fusion insights available")
        
        # Test radar insights
        print("\n📡 Radar Insights:")
        radar_insights = image_analyzer.analyze_radar_insights(scene_id)
        if radar_insights:
            coverage = radar_insights.get('radar_coverage', {})
            print(f"  • Total radar samples: {coverage.get('total_radar_samples', 0)}")
            print(f"  • Coverage rate: {coverage.get('coverage_rate', 0):.2%}")
            print(f"  • Average radars per sample: {coverage.get('average_radars_per_sample', 0):.1f}")
            print(f"  • Weather robustness: {radar_insights.get('weather_robustness', {}).get('weather_independence', False)}")
        else:
            print("  ❌ No radar insights available")
        
        # Test LiDAR insights
        print("\n💡 LiDAR Insights:")
        lidar_insights = image_analyzer.analyze_lidar_insights(scene_id)
        if lidar_insights:
            coverage = lidar_insights.get('lidar_coverage', {})
            print(f"  • Samples with LiDAR: {coverage.get('total_samples_with_lidar', 0)}")
            print(f"  • LiDAR coverage rate: {coverage.get('coverage_rate', 0):.2%}")
            print(f"  • 32-beam coverage: {coverage.get('32_beam_coverage', False)}")
            print(f"  • 3D mapping capability: {lidar_insights.get('spatial_insights', {}).get('3d_mapping_capability', False)}")
        else:
            print("  ❌ No LiDAR insights available")
        
        # Test environmental insights
        print("\n🌍 Environmental Insights:")
        env_insights = image_analyzer.analyze_environmental_insights(scene_id)
        if env_insights:
            env_understanding = env_insights.get('environmental_understanding', {})
            print(f"  • Weather conditions: {env_understanding.get('weather_conditions', [])}")
            print(f"  • Road conditions: {env_understanding.get('road_conditions', [])}")
            print(f"  • Lighting conditions: {env_understanding.get('lighting_conditions', [])}")
            print(f"  • Traffic conditions: {env_understanding.get('traffic_conditions', [])}")
            
            robustness = env_insights.get('sensor_robustness', {})
            print(f"  • Camera robustness: {robustness.get('camera_robustness', {}).get('daylight_performance', False)}")
            print(f"  • Radar robustness: {robustness.get('radar_robustness', {}).get('weather_robustness', False)}")
            print(f"  • LiDAR robustness: {robustness.get('lidar_robustness', {}).get('precision_mapping', False)}")
            print(f"  • Overall robustness score: {env_insights.get('overall_robustness_score', 0):.2f}")
        else:
            print("  ❌ No environmental insights available")
        
        print("\n✅ ImageAnalyzer test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error testing ImageAnalyzer: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_image_analyzer() 