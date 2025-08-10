#!/usr/bin/env python3
"""
Test script to verify analyzers are working correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analysis.data_loader import DataLoader
from analysis.qa_analyzer import QAAnalyzer
from analysis.vehicle_state_analyzer import VehicleStateAnalyzer
from analysis.image_analysis import ImageAnalyzer

def test_analyzers():
    """Test all analyzers with scene 1"""
    print("🧪 Testing Analyzers...")
    
    try:
        # Initialize data loader
        loader = DataLoader()
        print("✅ DataLoader initialized")
        
        # Test QA Analyzer
        qa_analyzer = QAAnalyzer(loader)
        qa_dist = qa_analyzer._get_qa_distribution(1, 0)
        print(f"✅ QA Analyzer working - QA distribution: {qa_dist}")
        
        # Test Vehicle State Analyzer
        vehicle_analyzer = VehicleStateAnalyzer(loader)
        scene_analysis = vehicle_analyzer.analyze_scene(1)
        print(f"✅ Vehicle State Analyzer working - Analysis keys: {list(scene_analysis.keys())}")
        
        # Test Image Analyzer
        image_analyzer = ImageAnalyzer(loader)
        visual_insights = image_analyzer.analyze_visual_insights(1)
        print(f"✅ Image Analyzer working - Visual insights: {bool(visual_insights)}")
        
        print("\n🎉 All analyzers are working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing analyzers: {e}")
        return False

if __name__ == "__main__":
    test_analyzers() 