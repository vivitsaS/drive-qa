"""
Main Analysis Script for DriveLM Dataset

Runs comprehensive analysis pipeline and generates reports.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from analysis.data_loader import DriveLMDataLoader
from analysis.question_analyzer import QuestionAnalyzer
from analysis.object_analyzer import ObjectAnalyzer
from analysis.spatial_analyzer import SpatialAnalyzer
from analysis.temporal_analyzer import TemporalAnalyzer
from analysis.safety_analyzer import SafetyAnalyzer
from analysis.multimodal_analyzer import MultimodalAnalyzer
from analysis.visualizer import AnalysisVisualizer


def main():
    """Main analysis pipeline"""
    print("Starting DriveLM Data Analysis...")
    
    # 1. Load all data
    print("Loading data...")
    loader = DriveLMDataLoader()
    all_data = loader.load_all_data()
    
    # 2. Initialize analyzers
    print("Initializing analyzers...")
    question_analyzer = QuestionAnalyzer()
    object_analyzer = ObjectAnalyzer()
    spatial_analyzer = SpatialAnalyzer()
    temporal_analyzer = TemporalAnalyzer()
    safety_analyzer = SafetyAnalyzer()
    multimodal_analyzer = MultimodalAnalyzer()
    
    # 3. Run analyses
    print("Running question analysis...")
    question_results = question_analyzer.classify_questions([])
    
    print("Running object analysis...")
    object_results = object_analyzer.analyze_object_distribution([])
    
    print("Running spatial analysis...")
    spatial_results = spatial_analyzer.analyze_relative_positions([])
    
    print("Running temporal analysis...")
    temporal_results = temporal_analyzer.analyze_temporal_questions([])
    
    print("Running safety analysis...")
    safety_results = safety_analyzer.identify_safety_questions([])
    
    print("Running multimodal analysis...")
    multimodal_results = multimodal_analyzer.analyze_question_image_mapping([], {})
    
    # 4. Create visualizations
    print("Creating visualizations...")
    visualizer = AnalysisVisualizer()
    visualizer.create_comprehensive_dashboard({
        'questions': question_results,
        'objects': object_results,
        'spatial': spatial_results,
        'temporal': temporal_results,
        'safety': safety_results,
        'multimodal': multimodal_results
    })
    
    # 5. Generate report
    print("Generating report...")
    generate_report()
    
    print("Analysis complete!")


def generate_report():
    """Generate analysis report"""
    pass


if __name__ == "__main__":
    main() 