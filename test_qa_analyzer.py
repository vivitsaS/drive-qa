#!/usr/bin/env python3

"""Test script for QuestionDistributionAnalyzer"""

from analysis.data_loader import DataLoader
from analysis.question_distribution_analyzer import QuestionDistributionAnalyzer

def test_qa_analyzer():
    """Test the QuestionDistributionAnalyzer with scene 1"""
    
    # Initialize
    loader = DataLoader("data/concatenated_data/concatenated_data.json")
    analyzer = QuestionDistributionAnalyzer(loader)
    
    print("🔍 Testing QuestionDistributionAnalyzer...")
    print("=" * 50)
    
    # Test comprehensive analysis
    analysis = analyzer.generate_comprehensive_qa_analysis(1)
    
    print("📊 QA Distribution Analysis:")
    print(f"Total QA pairs: {analysis['qa_distribution']['total_qa_pairs']}")
    print(f"Keyframe distribution: {analysis['qa_distribution']['keyframe_distribution']}")
    print(f"Functional split: {analysis['qa_distribution']['functional_split']}")
    print()
    
    print("📝 Question Categories:")
    for category, stats in analysis['question_categories']['category_distribution'].items():
        print(f"  {category}: {stats['count']} questions ({stats['percentage']:.1f}%)")
    print()
    
    print("🧠 Question Complexity:")
    complexity = analysis['question_complexity']
    print(f"  Single-step: {complexity['single_step_count']} ({complexity['single_step_percentage']:.1f}%)")
    print(f"  Multi-step: {complexity['multi_step_count']} ({complexity['multi_step_percentage']:.1f}%)")
    print(f"  Average complexity score: {complexity['average_complexity_score']:.3f}")
    print()
    
    print("💬 Answer Complexity:")
    answer_comp = analysis['answer_complexity']
    print(f"  Simple answers: {answer_comp['simple_answers']} ({answer_comp['simple_answer_percentage']:.1f}%)")
    print(f"  Detailed answers: {answer_comp['detailed_answers']} ({answer_comp['detailed_answer_percentage']:.1f}%)")
    print(f"  Average answer length: {answer_comp['average_answer_length']:.1f} words")
    print()
    
    print("🚗 Object Frequency:")
    obj_freq = analysis['object_frequency']
    print(f"  Most frequent object: {obj_freq['most_frequent_object']}")
    for obj, count in list(obj_freq['object_frequency'].items())[:5]:
        print(f"  {obj}: {count} questions")
    print()
    
    print("📍 Spatial Relationships:")
    spatial = analysis['spatial_relationships']
    print(f"  Total spatial questions: {spatial['total_spatial_questions']}")
    print(f"  Most common spatial type: {spatial['most_common_spatial_type']}")
    for spatial_type, count in spatial['spatial_distribution'].items():
        if count > 0:
            print(f"  {spatial_type}: {count} questions")
    print()
    
    print("🖼️ Image Correlation:")
    img_corr = analysis['image_correlation']
    print(f"  QA with images: {img_corr['qa_with_images']}")
    print(f"  QA without images: {img_corr['qa_without_images']}")
    print(f"  Image correlation: {img_corr['image_correlation_percentage']:.1f}%")
    print()
    
    print("✅ Test completed successfully!")

if __name__ == "__main__":
    test_qa_analyzer() 