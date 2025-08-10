#!/usr/bin/env python3

"""
DriveLM Analysis Dashboard

A comprehensive Streamlit dashboard for analyzing autonomous driving behavior
across the DriveLM dataset.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import json

from analysis.data_loader import DataLoader
from analysis.vehicle_state_analyzer import VehicleStateAnalyzer
from analysis.qa_analyzer import QAAnalyzer
from analysis.image_analysis import ImageAnalyzer


# Page configuration
st.set_page_config(
    page_title="DriveLM Analysis Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize data loader and analyzer
@st.cache_resource
def initialize_analyzers():
    """Initialize data loader and analyzers with caching"""
    loader = DataLoader("data/concatenated_data/concatenated_data.json")
    vehicle_analyzer = VehicleStateAnalyzer(loader)
    qa_analyzer = QAAnalyzer(loader)
    image_analyzer = ImageAnalyzer(loader)
    return loader, vehicle_analyzer, qa_analyzer, image_analyzer

# Load all scene data
@st.cache_data
def load_all_scenes():
    """Load all scene data with caching"""
    try:
        loader, vehicle_analyzer, qa_analyzer, image_analyzer = initialize_analyzers()
        all_data = loader.load_all_data()
        scene_tokens = list(all_data.keys())
        
        # Generate analysis for all scenes
        all_analyses = {}
        for i, scene_token in enumerate(scene_tokens, 1):
            try:
                analysis = vehicle_analyzer.analyze_scene(i)
                all_analyses[i] = analysis
            except Exception as e:
                st.error(f"Error analyzing scene {i}: {e}")
                # Add a placeholder analysis to prevent errors
                all_analyses[i] = {
                    'velocity_summary': {'avg_speed': 0, 'max_speed': 0, 'min_speed': 0, 'speed_std': 0, 'avg_acceleration': 0, 'max_acceleration': 0, 'total_distance': 0, 'total_duration': 0, 'movement_segments': {'turning': 0, 'straight': 0, 'stopping': 0}},
                    'risk_assessment': {'risk_score': 0, 'risk_level': 'low', 'speed_risk': 0, 'acceleration_risk': 0, 'jerk_risk': 0, 'max_speed': 0, 'max_acceleration': 0, 'max_jerk': 0},
                    'safety_margins': {'safety_score': 0, 'avg_safety_margin': 0, 'min_safety_margin': 0, 'close_interactions': 0, 'high_risk_interactions': 0},
                    'collision_risk': {'avg_collision_risk': 0, 'max_collision_risk': 0, 'high_risk_objects': 0},
                    'traffic_compliance': {'compliance_score': 0},
                    'smoothness': {'smoothness_score': 0},
                    'predictability': {'predictability_score': 0},
                    'driving_style': {'style': 'unknown', 'overall_score': 0},
                    'system_performance': {'system_health': 'good', 'total_issues': 0, 'high_severity_issues': 0, 'medium_severity_issues': 0, 'issues': []}
                }
        
        return all_analyses, scene_tokens
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return {}, []


def main():
    """Main dashboard function"""
    
    # Header
    st.title("🚗 Nuscenes-DriveLM Dataset Analysis Dashboard")
    st.markdown("---")
    
    # Initialize
    loader, vehicle_analyzer, qa_analyzer, image_analyzer = initialize_analyzers()
    all_analyses, scene_tokens = load_all_scenes()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Executive Summary", "Scene Analysis", "Camera & Sensor Analysis", "Comparative Analysis", "Risk Assessment", "Data Quality", "QA Analysis"]
    )
    
    # Page routing
    if page == "Executive Summary":
        show_executive_summary(all_analyses, scene_tokens)
    elif page == "Scene Analysis":
        show_scene_analysis(all_analyses, scene_tokens, vehicle_analyzer)
    elif page == "Camera & Sensor Analysis":
        show_camera_sensor_analysis(image_analyzer, scene_tokens)
    elif page == "Comparative Analysis":
        show_comparative_analysis(all_analyses, scene_tokens)
    elif page == "Risk Assessment":
        show_risk_assessment(all_analyses, scene_tokens)
    elif page == "Data Quality":
        show_data_quality(all_analyses, scene_tokens)
    elif page == "QA Analysis":
        show_qa_analysis(qa_analyzer, scene_tokens)


def show_executive_summary(all_analyses, scene_tokens):
    """Show executive summary dashboard"""
    
    st.header("📊 Executive Summary")
    st.markdown("High-level overview of the DriveLM dataset analysis")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Scenes", len(scene_tokens))
    
    with col2:
        risk_scores = [analysis['risk_assessment']['risk_score'] for analysis in all_analyses.values()]
        avg_risk = np.mean(risk_scores) if risk_scores else 0
        st.metric("Average Risk Score", f"{avg_risk:.3f}")
    
    with col3:
        safety_scores = [analysis['safety_margins']['safety_score'] for analysis in all_analyses.values()]
        avg_safety = np.mean(safety_scores) if safety_scores else 0
        st.metric("Average Safety Score", f"{avg_safety:.3f}")
    
    with col4:
        compliance_scores = [analysis['traffic_compliance']['compliance_score'] for analysis in all_analyses.values()]
        avg_compliance = np.mean(compliance_scores) if compliance_scores else 0
        st.metric("Average Compliance", f"{avg_compliance:.1%}")
    
    # Cross-scene comparison chart
    st.subheader("Cross-Scene Performance Comparison")
    
    # Prepare data for comparison
    comparison_data = []
    for scene_id, analysis in all_analyses.items():
        try:
                    comparison_data.append({
            'Scene': f"Scene {scene_id}",
            'Risk Score': analysis.get('risk_assessment', {}).get('risk_score', 0),
            'Safety Score': analysis.get('safety_margins', {}).get('safety_score', 0),
            'Compliance Score': analysis.get('traffic_compliance', {}).get('compliance_score', 0),
            'Smoothness Score': analysis.get('smoothness', {}).get('smoothness_score', 0),
            'Predictability Score': analysis.get('predictability', {}).get('predictability_score', 0),
            'Driving Style Score': analysis.get('driving_style', {}).get('overall_score', 0)
        })
        except KeyError as e:
            st.warning(f"Missing data for Scene {scene_id}: {e}")
            continue
    
    if not comparison_data:
        st.error("No comparison data available")
        return
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Radar chart for performance metrics
    if len(df_comparison) > 0:
        fig_radar = go.Figure()
        
        for _, row in df_comparison.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['Risk Score'], row['Safety Score'], row['Compliance Score'], 
                   row['Smoothness Score'], row['Predictability Score'], row['Driving Style Score']],
                theta=['Risk', 'Safety', 'Compliance', 'Smoothness', 'Predictability', 'Style'],
                fill='toself',
                name=row['Scene']
            ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Performance Radar Chart"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("No data available for radar chart")
    
    # Summary statistics
    st.subheader("Summary Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Driving style distribution
        style_counts = {}
        for analysis in all_analyses.values():
            style = analysis['driving_style']['style']
            style_counts[style] = style_counts.get(style, 0) + 1
        
        fig_style = px.pie(
            values=list(style_counts.values()),
            names=list(style_counts.keys()),
            title="Driving Style Distribution"
        )
        st.plotly_chart(fig_style, use_container_width=True)
    
    with col2:
        # Risk level distribution
        risk_counts = {}
        for analysis in all_analyses.values():
            risk_level = analysis['risk_assessment']['risk_level']
            risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
        
        if risk_counts:
            risk_df = pd.DataFrame({
                'Risk Level': list(risk_counts.keys()),
                'Count': list(risk_counts.values())
            })
            fig_risk = px.bar(
                risk_df,
                x='Risk Level',
                y='Count',
                title="Risk Level Distribution"
            )
        else:
            fig_risk = px.bar(
                pd.DataFrame({'Risk Level': ['No Data'], 'Count': [0]}),
                x='Risk Level',
                y='Count',
                title="Risk Level Distribution"
            )
        st.plotly_chart(fig_risk, use_container_width=True)


def show_scene_analysis(all_analyses, scene_tokens, analyzer):
    """Show detailed scene analysis"""
    
    st.header("🔍 Scene Analysis")
    
    # Scene selector
    selected_scene = st.selectbox("Select Scene", range(1, len(scene_tokens) + 1))
    
    if selected_scene in all_analyses:
        analysis = all_analyses[selected_scene]
        
        # Scene information
        st.subheader(f"Scene {selected_scene} Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Average Speed", f"{analysis['velocity_summary']['avg_speed']:.2f} m/s")
            st.metric("Max Speed", f"{analysis['velocity_summary']['max_speed']:.2f} m/s")
            st.metric("Total Distance", f"{analysis['velocity_summary']['total_distance']:.2f} m")
            st.metric("Duration", f"{analysis['velocity_summary']['total_duration']:.2f} s")
        
        with col2:
            st.metric("Risk Level", analysis['risk_assessment']['risk_level'].upper())
            st.metric("Safety Score", f"{analysis['safety_margins']['safety_score']:.3f}")
            st.metric("Compliance", f"{analysis['traffic_compliance']['compliance_score']:.1%}")
            st.metric("Smoothness", f"{analysis['smoothness']['smoothness_score']:.3f}")
        
        # Detailed metrics
        st.subheader("Detailed Metrics")
        
        # Create tabs for different analysis types
        tab1, tab2, tab3, tab4 = st.tabs(["Movement Analysis", "Safety Analysis", "Risk Analysis", "Quality Analysis"])
        
        with tab1:
            show_movement_analysis(analysis)
        
        with tab2:
            show_safety_analysis(analysis)
        
        with tab3:
            show_risk_analysis(analysis)
        
        with tab4:
            show_quality_analysis(analysis)


def show_movement_analysis(analysis):
    """Show movement analysis details"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Speed metrics
        st.write("**Speed Analysis**")
        speed_data = {
            'Metric': ['Average', 'Maximum', 'Minimum', 'Std Dev'],
            'Speed (m/s)': [
                analysis['velocity_summary']['avg_speed'],
                analysis['velocity_summary']['max_speed'],
                analysis['velocity_summary']['min_speed'],
                analysis['velocity_summary']['speed_std']
            ]
        }
        st.dataframe(pd.DataFrame(speed_data))
    
    with col2:
        # Acceleration metrics
        st.write("**Acceleration Analysis**")
        accel_data = {
            'Metric': ['Average', 'Maximum'],
            'Acceleration (m/s²)': [
                analysis['velocity_summary']['avg_acceleration'],
                analysis['velocity_summary']['max_acceleration']
            ]
        }
        st.dataframe(pd.DataFrame(accel_data))
    
    # Movement segments
    st.write("**Movement Segments**")
    segments_data = {
        'Segment Type': ['Turning', 'Straight', 'Stopping'],
        'Count': [
            str(analysis['velocity_summary']['movement_segments']['turning']),
            str(analysis['velocity_summary']['movement_segments']['straight']),
            str(analysis['velocity_summary']['movement_segments']['stopping'])
        ]
    }
    st.dataframe(pd.DataFrame(segments_data))


def show_safety_analysis(analysis):
    """Show safety analysis details"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Safety margins
        st.write("**Safety Margins**")
        safety_data = {
            'Metric': ['Average', 'Minimum', 'Close Interactions', 'High Risk'],
            'Value': [
                f"{analysis['safety_margins']['avg_safety_margin']:.2f} m",
                f"{analysis['safety_margins']['min_safety_margin']:.2f} m",
                str(analysis['safety_margins']['close_interactions']),
                str(analysis['safety_margins']['high_risk_interactions'])
            ]
        }
        st.dataframe(pd.DataFrame(safety_data))
    
    with col2:
        # Collision risk
        st.write("**Collision Risk**")
        collision_data = {
            'Metric': ['Average Risk', 'Max Risk', 'High Risk Objects'],
            'Value': [
                f"{analysis['collision_risk']['avg_collision_risk']:.4f}",
                f"{analysis['collision_risk']['max_collision_risk']:.4f}",
                str(analysis['collision_risk']['high_risk_objects'])
            ]
        }
        st.dataframe(pd.DataFrame(collision_data))


def show_risk_analysis(analysis):
    """Show risk analysis details"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk components
        st.write("**Risk Components**")
        risk_data = {
            'Component': ['Speed Risk', 'Acceleration Risk', 'Jerk Risk'],
            'Score': [
                analysis['risk_assessment']['speed_risk'],
                analysis['risk_assessment']['acceleration_risk'],
                analysis['risk_assessment']['jerk_risk']
            ]
        }
        st.dataframe(pd.DataFrame(risk_data))
    
    with col2:
        # Risk metrics
        st.write("**Risk Metrics**")
        metrics_data = {
            'Metric': ['Max Speed', 'Max Acceleration', 'Max Jerk'],
            'Value': [
                f"{analysis['risk_assessment']['max_speed']:.2f} m/s",
                f"{analysis['risk_assessment']['max_acceleration']:.2f} m/s²",
                f"{analysis['risk_assessment']['max_jerk']:.2f} m/s³"
            ]
        }
        st.dataframe(pd.DataFrame(metrics_data))


def show_quality_analysis(analysis):
    """Show data quality analysis details"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Quality metrics
        st.write("**Data Quality Metrics**")
        quality_data = {
            'Metric': ['Completeness', 'Temporal Consistency', 'Annotation Correlation'],
            'Score': [
                f"{analysis['data_quality']['completeness_rate']:.1%}",
                "✓" if analysis['data_quality']['temporal_consistency'] else "✗",
                f"{analysis['data_quality']['annotation_correlation_rate']:.1%}"
            ]
        }
        st.dataframe(pd.DataFrame(quality_data))
    
    with col2:
        # Sensor reliability
        st.write("**Sensor Reliability**")
        sensor_data = []
        for sensor, reliability in analysis['data_quality']['sensor_reliability'].items():
            sensor_data.append([str(sensor), f"{reliability:.1%}"])
        
        st.dataframe(pd.DataFrame(sensor_data, columns=['Sensor', 'Reliability']))


def show_comparative_analysis(all_analyses, scene_tokens):
    """Show comparative analysis across scenes"""
    
    st.header("📈 Comparative Analysis")
    
    # Prepare comparison data
    comparison_data = []
    for scene_id, analysis in all_analyses.items():
        comparison_data.append({
            'Scene': f"Scene {scene_id}",
            'Risk Score': analysis['risk_assessment']['risk_score'],
            'Safety Score': analysis['safety_margins']['safety_score'],
            'Compliance Score': analysis['traffic_compliance']['compliance_score'],
            'Smoothness Score': analysis['smoothness']['smoothness_score'],
            'Predictability Score': analysis['predictability']['predictability_score'],
            'Driving Style': analysis['driving_style']['style'],
            'Risk Level': analysis['risk_assessment']['risk_level']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk vs Safety scatter plot
        fig_scatter = px.scatter(
            df_comparison,
            x='Risk Score',
            y='Safety Score',
            color='Driving Style',
            size='Compliance Score',
            hover_data=['Scene'],
            title="Risk vs Safety Analysis"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Performance metrics bar chart
        metrics = ['Risk Score', 'Safety Score', 'Compliance Score', 'Smoothness Score', 'Predictability Score']
        fig_bar = px.bar(
            df_comparison,
            x='Scene',
            y=metrics,
            title="Performance Metrics Comparison",
            barmode='group'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Detailed comparison table
    st.subheader("Detailed Comparison Table")
    st.dataframe(df_comparison)


def show_risk_assessment(all_analyses, scene_tokens):
    """Show risk assessment dashboard"""
    
    st.header("⚠️ Risk Assessment Dashboard")
    
        # Risk overview
    col1, col2, col3 = st.columns(3)
    
    high_risk_scenes = sum(1 for analysis in all_analyses.values()
                          if analysis.get('risk_assessment', {}).get('risk_level') == 'high')
    medium_risk_scenes = sum(1 for analysis in all_analyses.values()
                            if analysis.get('risk_assessment', {}).get('risk_level') == 'medium')
    low_risk_scenes = sum(1 for analysis in all_analyses.values()
                         if analysis.get('risk_assessment', {}).get('risk_level') == 'low')
    
    with col1:
        st.metric("High Risk Scenes", high_risk_scenes, delta=None)
    
    with col2:
        st.metric("Medium Risk Scenes", medium_risk_scenes, delta=None)
    
    with col3:
        st.metric("Low Risk Scenes", low_risk_scenes, delta=None)
    
    # Risk distribution
    st.subheader("Risk Distribution")
    
    risk_data = []
    for scene_id, analysis in all_analyses.items():
        risk_data.append({
            'Scene': f"Scene {scene_id}",
            'Risk Score': analysis['risk_assessment']['risk_score'],
            'Risk Level': analysis['risk_assessment']['risk_level'],
            'Speed Risk': analysis['risk_assessment']['speed_risk'],
            'Acceleration Risk': analysis['risk_assessment']['acceleration_risk'],
            'Jerk Risk': analysis['risk_assessment']['jerk_risk']
        })
    
    df_risk = pd.DataFrame(risk_data)
    
    # Risk heatmap
    fig_heatmap = px.imshow(
        df_risk[['Speed Risk', 'Acceleration Risk', 'Jerk Risk']].T,
        labels=dict(x="Scene", y="Risk Type", color="Risk Score"),
        title="Risk Component Heatmap"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Risk details table
    st.subheader("Risk Details")
    st.dataframe(df_risk)


def show_data_quality(all_analyses, scene_tokens):
    """Show data quality dashboard"""
    
    st.header("🔍 Data Quality Dashboard")
    
    # Quality overview
    col1, col2, col3 = st.columns(3)
    
    # Since data_quality is not available, use system_performance as proxy
    avg_quality = np.mean([1.0 if analysis.get('system_performance', {}).get('system_health') == 'good' else 0.5
                          for analysis in all_analyses.values()])
    avg_completeness = np.mean([1.0 - (analysis.get('system_performance', {}).get('total_issues', 0) / 10.0)
                               for analysis in all_analyses.values()])
    total_issues = sum(len(analysis.get('system_performance', {}).get('issues', [])) 
                       for analysis in all_analyses.values())
    
    with col1:
        st.metric("Average Quality Score", f"{avg_quality:.3f}")
    
    with col2:
        st.metric("Average Completeness", f"{avg_completeness:.1%}")
    
    with col3:
        st.metric("Total Issues", total_issues)
    
    # Quality metrics by scene
    st.subheader("Quality Metrics by Scene")
    
    quality_data = []
    for scene_id, analysis in all_analyses.items():
        system_perf = analysis.get('system_performance', {})
        quality_data.append({
            'Scene': f"Scene {scene_id}",
            'Overall Quality': 1.0 if system_perf.get('system_health') == 'good' else 0.5,
            'Completeness': 1.0 - (system_perf.get('total_issues', 0) / 10.0),
            'Temporal Consistency': 1.0 if system_perf.get('high_severity_issues', 0) == 0 else 0.5,
            'Annotation Correlation': 1.0 - (system_perf.get('medium_severity_issues', 0) / 5.0),
            'System Health': system_perf.get('system_health', 'unknown')
        })
    
    df_quality = pd.DataFrame(quality_data)
    
    # Quality metrics chart
    fig_quality = px.bar(
        df_quality,
        x='Scene',
        y=['Overall Quality', 'Completeness', 'Annotation Correlation'],
        title="Data Quality Metrics by Scene",
        barmode='group'
    )
    st.plotly_chart(fig_quality, use_container_width=True)
    
    # Quality details table
    st.subheader("Quality Details")
    st.dataframe(df_quality)


def show_camera_sensor_analysis(image_analyzer, scene_tokens):
    """Show camera and sensor analysis insights"""
    
    st.header("📷 Camera & Sensor Analysis")
    st.markdown("Insights learned from nuScenes sensor suite")
    
    # Scene selection
    selected_scene = st.selectbox("Select Scene", range(1, len(scene_tokens) + 1))
    
    if st.button("Analyze Scene"):
        with st.spinner("Analyzing camera and sensor data..."):
            
            # Visual Insights
            st.subheader("🔍 Visual Insights")
            visual_insights = image_analyzer.analyze_visual_insights(selected_scene)
            
            if visual_insights:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Objects Detected", visual_insights.get('object_detection_insights', {}).get('total_objects_detected', 0))
                
                with col2:
                    st.metric("Object Categories", visual_insights.get('object_detection_insights', {}).get('unique_object_categories', 0))
                
                with col3:
                    traffic_density = visual_insights.get('scene_understanding', {}).get('traffic_density', 'unknown')
                    st.metric("Traffic Density", traffic_density.title())
                
                with col4:
                    env_complexity = visual_insights.get('scene_understanding', {}).get('environment_complexity', 'unknown')
                    st.metric("Environment Complexity", env_complexity.title())
                
                # Camera coverage analysis
                st.subheader("📹 Camera Coverage Analysis")
                camera_coverage = visual_insights.get('camera_coverage_analysis', {})
                
                if camera_coverage.get('objects_per_camera'):
                    camera_data = pd.DataFrame([
                        {'Camera': cam, 'Objects': count} 
                        for cam, count in camera_coverage['objects_per_camera'].items()
                    ])
                    
                    fig = px.bar(camera_data, x='Camera', y='Objects', 
                               title="Objects Detected per Camera",
                               color='Objects', color_continuous_scale='viridis')
                    st.plotly_chart(fig, use_container_width=True)
            
            # Comprehensive Sensor Fusion
            st.subheader("🔗 Comprehensive Sensor Fusion")
            fusion_insights = image_analyzer.analyze_comprehensive_sensor_fusion(selected_scene)
            
            if fusion_insights:
                col1, col2, col3, col4 = st.columns(4)
                
                availability = fusion_insights.get('sensor_availability', {})
                with col1:
                    st.metric("Total Samples", availability.get('total_samples', 0))
                
                with col2:
                    coverage_rate = availability.get('full_suite_coverage_rate', 0)
                    st.metric("Full Suite Coverage", f"{coverage_rate:.1%}")
                
                with col3:
                    avg_cameras = availability.get('average_cameras_per_sample', 0)
                    st.metric("Avg Cameras/Sample", f"{avg_cameras:.1f}")
                
                with col4:
                    avg_radar = availability.get('average_radar_per_sample', 0)
                    st.metric("Avg Radar/Sample", f"{avg_radar:.1f}")
                
                # 360° Coverage Analysis
                st.subheader("🌐 360° Coverage Analysis")
                coverage = fusion_insights.get('360_degree_coverage', {})
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Visual Coverage", "✅" if coverage.get('visual_coverage') else "❌")
                with col2:
                    st.metric("Radar Coverage", "✅" if coverage.get('radar_coverage') else "❌")
                with col3:
                    st.metric("LiDAR Coverage", "✅" if coverage.get('lidar_coverage') else "❌")
                with col4:
                    st.metric("Comprehensive", "✅" if coverage.get('comprehensive_coverage') else "❌")
            
            # Radar Insights
            st.subheader("📡 Radar Insights")
            radar_insights = image_analyzer.analyze_radar_insights(selected_scene)
            
            if radar_insights:
                col1, col2, col3, col4 = st.columns(4)
                
                coverage = radar_insights.get('radar_coverage', {})
                with col1:
                    st.metric("Radar Samples", coverage.get('total_radar_samples', 0))
                
                with col2:
                    coverage_rate = coverage.get('coverage_rate', 0)
                    st.metric("Coverage Rate", f"{coverage_rate:.1%}")
                
                with col3:
                    avg_radars = coverage.get('average_radars_per_sample', 0)
                    st.metric("Avg Radars/Sample", f"{avg_radars:.1f}")
                
                with col4:
                    weather_robust = radar_insights.get('weather_robustness', {}).get('weather_independence', False)
                    st.metric("Weather Robust", "✅" if weather_robust else "❌")
            
            # LiDAR Insights
            st.subheader("💡 LiDAR Insights")
            lidar_insights = image_analyzer.analyze_lidar_insights(selected_scene)
            
            if lidar_insights:
                col1, col2, col3, col4 = st.columns(4)
                
                coverage = lidar_insights.get('lidar_coverage', {})
                with col1:
                    st.metric("LiDAR Samples", coverage.get('total_samples_with_lidar', 0))
                
                with col2:
                    coverage_rate = coverage.get('coverage_rate', 0)
                    st.metric("Coverage Rate", f"{coverage_rate:.1%}")
                
                with col3:
                    beam_coverage = coverage.get('32_beam_coverage', False)
                    st.metric("32-Beam Coverage", "✅" if beam_coverage else "❌")
                
                with col4:
                    mapping_3d = lidar_insights.get('spatial_insights', {}).get('3d_mapping_capability', False)
                    st.metric("3D Mapping", "✅" if mapping_3d else "❌")
            
            # Environmental Insights
            st.subheader("🌍 Environmental Insights")
            env_insights = image_analyzer.analyze_environmental_insights(selected_scene)
            
            if env_insights:
                env_understanding = env_insights.get('environmental_understanding', {})
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    weather = env_understanding.get('weather_conditions', [])
                    st.metric("Weather", weather[0] if weather else "Clear")
                
                with col2:
                    road = env_understanding.get('road_conditions', [])
                    st.metric("Road Conditions", road[0] if road else "Normal")
                
                with col3:
                    lighting = env_understanding.get('lighting_conditions', [])
                    st.metric("Lighting", lighting[0] if lighting else "Daylight")
                
                with col4:
                    robustness_score = env_insights.get('overall_robustness_score', 0)
                    st.metric("Robustness Score", f"{robustness_score:.2f}")
                
                # Sensor Robustness Analysis
                st.subheader("🛡️ Sensor Robustness")
                robustness = env_insights.get('sensor_robustness', {})
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    camera_robust = robustness.get('camera_robustness', {})
                    st.write("**Camera Robustness:**")
                    st.write(f"• Daylight: {'✅' if camera_robust.get('daylight_performance') else '❌'}")
                    st.write(f"• Low Light: {'✅' if camera_robust.get('low_light_performance') else '❌'}")
                    st.write(f"• Weather: {'✅' if camera_robust.get('visual_weather_impact') else '❌'}")
                
                with col2:
                    radar_robust = robustness.get('radar_robustness', {})
                    st.write("**Radar Robustness:**")
                    st.write(f"• Weather: {'✅' if radar_robust.get('weather_robustness') else '❌'}")
                    st.write(f"• Velocity: {'✅' if radar_robust.get('velocity_detection') else '❌'}")
                    st.write(f"• Distance: {'✅' if radar_robust.get('distance_accuracy') else '❌'}")
                
                with col3:
                    lidar_robust = robustness.get('lidar_robustness', {})
                    st.write("**LiDAR Robustness:**")
                    st.write(f"• 3D Mapping: {'✅' if lidar_robust.get('precision_mapping') else '❌'}")
                    st.write(f"• Weather: {'✅' if not lidar_robust.get('weather_sensitivity') else '❌'}")
                    st.write(f"• Resolution: {'✅' if lidar_robust.get('spatial_resolution') else '❌'}")

def show_qa_analysis(qa_analyzer, scene_tokens):
    """Show QA analysis dashboard"""
    
    st.header("❓ QA Analysis Dashboard")
    
    # Scene selector
    selected_scene = st.selectbox("Select Scene for QA Analysis", range(1, len(scene_tokens) + 1))
    
    if st.button("Analyze QA Distribution"):
        with st.spinner("Analyzing QA data..."):
            # Get QA distribution for the scene
            qa_distribution = qa_analyzer._get_qa_distribution(selected_scene, 0)  # 0 = all keyframes
            
            if qa_distribution:
                # QA Distribution Overview
                st.subheader("📊 QA Distribution Overview")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total QA Pairs", qa_distribution.get('total', 0))
                
                with col2:
                    st.metric("Perception QA", qa_distribution.get('perception', 0))
                
                with col3:
                    st.metric("Planning QA", qa_distribution.get('planning', 0))
                
                with col4:
                    st.metric("Prediction QA", qa_distribution.get('prediction', 0))
                
                # Functional Split Chart
                st.subheader("🧠 Functional Split")
                
                functional_split = {
                    'Perception': qa_distribution.get('perception', 0),
                    'Planning': qa_distribution.get('planning', 0),
                    'Prediction': qa_distribution.get('prediction', 0),
                    'Behavior': qa_distribution.get('behavior', 0)
                }
                
                fig_functional = px.pie(
                    values=list(functional_split.values()),
                    names=list(functional_split.keys()),
                    title="Question Types by Function"
                )
                st.plotly_chart(fig_functional, use_container_width=True)
            
            else:
                st.error("No QA data available for this scene")


if __name__ == "__main__":
    main() 