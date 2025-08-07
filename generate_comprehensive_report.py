#!/usr/bin/env python3

"""
Comprehensive Report Generator for DriveLM Dataset

Generates detailed written reports for all scenes in the concatenated JSON.
"""

import json
from datetime import datetime
from analysis.data_loader import DataLoader
from analysis.vehicle_state_analyzer import VehicleStateAnalyzer


def generate_scene_report(analyzer, scene_id, scene_data):
    """Generate a detailed report for a single scene"""
    
    # Get comprehensive analysis
    analysis = analyzer.generate_comprehensive_analysis(scene_id)
    
    # Extract scene info from the original scene data
    scene_info = {
        'scene_name': scene_data['scene_name'],
        'scene_description': scene_data['scene_description'],
        'nbr_samples': scene_data['nbr_samples']
    }
    velocity_summary = analysis['velocity_summary']
    driving_style = analysis['driving_style']
    smoothness = analysis['smoothness']
    predictability = analysis['predictability']
    risk_assessment = analysis['risk_assessment']
    safety_margins = analysis['safety_margins']
    collision_risk = analysis['collision_risk']
    traffic_compliance = analysis['traffic_compliance']
    system_performance = analysis['system_performance']
    event_correlation = analysis['event_correlation']
    data_quality = analysis['data_quality']
    
    report = f"""
{'='*80}
SCENE {scene_id} ANALYSIS REPORT
{'='*80}

SCENE INFORMATION:
- Scene Name: {scene_info['scene_name']}
- Scene Description: {scene_info['scene_description']}
- Number of Samples: {scene_info['nbr_samples']}

{'='*50}
VELOCITY AND MOVEMENT SUMMARY:
{'='*50}
- Average Speed: {velocity_summary['avg_speed']:.2f} m/s ({velocity_summary['avg_speed']*3.6:.1f} km/h)
- Maximum Speed: {velocity_summary['max_speed']:.2f} m/s ({velocity_summary['max_speed']*3.6:.1f} km/h)
- Minimum Speed: {velocity_summary['min_speed']:.2f} m/s ({velocity_summary['min_speed']*3.6:.1f} km/h)
- Speed Standard Deviation: {velocity_summary['speed_std']:.2f} m/s
- Average Acceleration: {velocity_summary['avg_acceleration']:.2f} m/s²
- Maximum Acceleration: {velocity_summary['max_acceleration']:.2f} m/s²
- Total Distance Traveled: {velocity_summary['total_distance']:.2f} meters
- Total Duration: {velocity_summary['total_duration']:.2f} seconds
- Movement Segments:
  * Turning Segments: {velocity_summary['movement_segments']['turning']}
  * Straight Segments: {velocity_summary['movement_segments']['straight']}
  * Stopping Periods: {velocity_summary['movement_segments']['stopping']}

{'='*50}
DRIVING STYLE ANALYSIS:
{'='*50}
- Overall Style: {driving_style['style'].upper()}
- Style Score: {driving_style['overall_score']:.3f} (0=conservative, 1=aggressive)
- Speed Component: {driving_style['speed_score']:.3f}
- Acceleration Component: {driving_style['acceleration_score']:.3f}
- Curvature Component: {driving_style['curvature_score']:.3f}
- Average Speed: {driving_style['metrics']['avg_speed']:.2f} m/s
- Maximum Speed: {driving_style['metrics']['max_speed']:.2f} m/s
- Average Acceleration: {driving_style['metrics']['avg_acceleration']:.2f} m/s²
- Maximum Acceleration: {driving_style['metrics']['max_acceleration']:.2f} m/s²
- Average Curvature: {driving_style['metrics']['avg_curvature']:.4f}

{'='*50}
SMOOTHNESS ANALYSIS:
{'='*50}
- Smoothness Level: {smoothness['smoothness_level'].upper()}
- Smoothness Score: {smoothness['smoothness_score']:.3f} (0=rough, 1=smooth)
- Average Jerk: {smoothness['avg_jerk']:.2f} m/s³
- Maximum Jerk: {smoothness['max_jerk']:.2f} m/s³
- Average Angular Acceleration: {smoothness['avg_angular_acceleration']:.4f} rad/s²
- Maximum Angular Acceleration: {smoothness['max_angular_acceleration']:.4f} rad/s²

{'='*50}
PREDICTABILITY ANALYSIS:
{'='*50}
- Predictability Level: {predictability['predictability_level'].upper()}
- Predictability Score: {predictability['predictability_score']:.3f} (0=unpredictable, 1=predictable)
- Speed Consistency: {predictability['speed_consistency']:.3f}
- Acceleration Consistency: {predictability['acceleration_consistency']:.3f}
- Curvature Consistency: {predictability['curvature_consistency']:.3f}

{'='*50}
RISK ASSESSMENT:
{'='*50}
- Overall Risk Level: {risk_assessment['risk_level'].upper()}
- Risk Score: {risk_assessment['risk_score']:.3f} (0=low, 1=high)
- Speed Risk: {risk_assessment['speed_risk']:.3f}
- Acceleration Risk: {risk_assessment['acceleration_risk']:.3f}
- Jerk Risk: {risk_assessment['jerk_risk']:.3f}
- Maximum Speed: {risk_assessment['max_speed']:.2f} m/s
- Maximum Acceleration: {risk_assessment['max_acceleration']:.2f} m/s²
- Maximum Jerk: {risk_assessment['max_jerk']:.2f} m/s³

{'='*50}
SAFETY ANALYSIS:
{'='*50}
- Average Safety Margin: {safety_margins['avg_safety_margin']:.2f} meters
- Minimum Safety Margin: {safety_margins['min_safety_margin']:.2f} meters
- Close Interactions (<5m): {safety_margins['close_interactions']}
- High Risk Interactions (<2m): {safety_margins['high_risk_interactions']}
- Safety Score: {safety_margins['safety_score']:.3f} (0=unsafe, 1=safe)

{'='*50}
COLLISION RISK ASSESSMENT:
{'='*50}
- Collision Risk Level: {collision_risk['risk_level'].upper()}
- Average Collision Risk: {collision_risk['avg_collision_risk']:.4f}
- Maximum Collision Risk: {collision_risk['max_collision_risk']:.4f}
- High Risk Objects (>50% risk): {collision_risk['high_risk_objects']}

{'='*50}
TRAFFIC COMPLIANCE:
{'='*50}
- Compliance Level: {traffic_compliance['compliance_level'].upper()}
- Overall Compliance Score: {traffic_compliance['compliance_score']:.3f}
- Speed Compliance Rate: {traffic_compliance['speed_compliance_rate']:.1%}
- Acceleration Compliance Rate: {traffic_compliance['acceleration_compliance_rate']:.1%}
- Speed Violations: {traffic_compliance['speed_violations']}
- Acceleration Violations: {traffic_compliance['acceleration_violations']}

{'='*50}
SYSTEM PERFORMANCE:
{'='*50}
- System Health: {system_performance['system_health'].upper()}
- Total Issues: {system_performance['total_issues']}
- High Severity Issues: {system_performance['high_severity_issues']}
- Medium Severity Issues: {system_performance['medium_severity_issues']}

{'='*50}
EVENT CORRELATION:
{'='*50}
- Total Events: {event_correlation['total_events']}
- Events with Images: {event_correlation['events_with_images']}
- Event Types: {dict(event_correlation['event_types'])}

{'='*50}
DATA QUALITY ASSESSMENT:
{'='*50}
- Overall Quality Score: {data_quality['overall_quality_score']:.3f} (0=poor, 1=excellent)
- Data Completeness Rate: {data_quality['completeness_rate']:.1%}
- Temporal Consistency: {'✓' if data_quality['temporal_consistency'] else '✗'}
- Annotation Correlation Rate: {data_quality['annotation_correlation_rate']:.1%}
- Sensor Reliability:
"""
    
    for sensor, reliability in data_quality['sensor_reliability'].items():
        report += f"  * {sensor}: {reliability:.1%}\n"
    
    report += f"""
- Missing Data Issues: {len(data_quality['missing_data_issues'])}

{'='*50}
SUMMARY AND RECOMMENDATIONS:
{'='*50}
"""
    
    # Generate summary and recommendations
    summary_points = []
    
    # Driving style summary
    if driving_style['style'] == 'aggressive':
        summary_points.append("• The vehicle exhibits aggressive driving behavior with high speeds and acceleration")
    elif driving_style['style'] == 'conservative':
        summary_points.append("• The vehicle shows conservative driving behavior with moderate speeds and smooth movements")
    else:
        summary_points.append("• The vehicle demonstrates moderate driving behavior with balanced speed and acceleration")
    
    # Risk assessment
    if risk_assessment['risk_level'] == 'high':
        summary_points.append("• HIGH RISK: The vehicle operates at dangerous speeds and acceleration levels")
    elif risk_assessment['risk_level'] == 'medium':
        summary_points.append("• MEDIUM RISK: The vehicle shows some concerning behavior patterns")
    else:
        summary_points.append("• LOW RISK: The vehicle operates within safe parameters")
    
    # Safety assessment
    if safety_margins['safety_score'] < 0.5:
        summary_points.append("• SAFETY CONCERN: Multiple close interactions with objects detected")
    else:
        summary_points.append("• GOOD SAFETY: Maintains appropriate distances from objects")
    
    # Compliance assessment
    if traffic_compliance['compliance_level'] == 'good':
        summary_points.append("• GOOD COMPLIANCE: Generally follows traffic rules with minor violations")
    else:
        summary_points.append("• COMPLIANCE ISSUES: Multiple traffic rule violations detected")
    
    # Data quality
    if data_quality['overall_quality_score'] > 0.9:
        summary_points.append("• EXCELLENT DATA QUALITY: All sensors functioning properly with complete data")
    else:
        summary_points.append("• DATA QUALITY ISSUES: Some sensors or data missing")
    
    # System performance
    if system_performance['system_health'] == 'good':
        summary_points.append("• GOOD SYSTEM HEALTH: No performance issues detected")
    else:
        summary_points.append("• SYSTEM ISSUES: Performance problems detected")
    
    for point in summary_points:
        report += point + "\n"
    
    report += f"""

{'='*80}
"""
    
    return report


def generate_comprehensive_report():
    """Generate comprehensive report for all scenes"""
    
    # Initialize data loader and analyzer
    loader = DataLoader("data/concatenated_data/concatenated_data.json")
    analyzer = VehicleStateAnalyzer(loader)
    
    # Get all scene data
    all_data = loader.load_all_data()
    scene_tokens = list(all_data.keys())
    
    # Generate report header
    report = f"""
{'='*100}
COMPREHENSIVE DRIVELM DATASET ANALYSIS REPORT
{'='*100}
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Scenes Analyzed: {len(scene_tokens)}

This report provides detailed analysis of autonomous driving behavior across all scenes
in the DriveLM dataset, including driving style, safety, risk assessment, and system performance.

{'='*100}
"""
    
    # Generate report for each scene
    for i, scene_token in enumerate(scene_tokens, 1):
        scene_data = all_data[scene_token]
        scene_report = generate_scene_report(analyzer, i, scene_data)
        report += scene_report
    
    # Generate overall summary
    report += f"""
{'='*100}
OVERALL DATASET SUMMARY
{'='*100}

Dataset Overview:
- Total Scenes: {len(scene_tokens)}
- Analysis Period: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Data Source: concatenated_data.json

Key Findings:
• This dataset contains {len(scene_tokens)} diverse driving scenarios
• Each scene provides comprehensive multimodal data including camera images, sensor data, and QA annotations
• The analysis reveals varying levels of driving complexity, risk, and system performance across scenarios

Recommendations for Further Analysis:
1. Cross-scene comparison to identify patterns in driving behavior
2. Temporal analysis to understand how driving patterns evolve
3. Correlation analysis between QA data and driving behavior
4. Performance benchmarking against human driving standards
5. Safety validation and risk mitigation strategies

{'='*100}
END OF REPORT
{'='*100}
"""
    
    return report


if __name__ == "__main__":
    # Generate the comprehensive report
    report = generate_comprehensive_report()
    
    # Save to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"comprehensive_analysis_report_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write(report)
    
    print(f"Comprehensive report generated and saved to: {filename}")
    print(f"Report length: {len(report)} characters")
    
    # Also print a summary to console
    print("\n" + "="*80)
    print("REPORT GENERATION COMPLETE")
    print("="*80)
    print("The comprehensive analysis report has been generated with detailed")
    print("insights for all scenes in the DriveLM dataset.")
    print("\nKey sections included:")
    print("• Velocity and movement analysis")
    print("• Driving style classification")
    print("• Safety and risk assessment")
    print("• Traffic compliance analysis")
    print("• System performance evaluation")
    print("• Data quality assessment")
    print("• Event correlation analysis")
    print("\nCheck the generated file for complete details.") 