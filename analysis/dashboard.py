import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys
import os

# Add the parent directory to the path so we can import from analysis
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.qa_analyzer import QAAnalyzer
from analysis.data_loader import DataLoader
from analysis.vehicle_state_analyzer import VehicleStateAnalyzer

def main():
    st.title("DriveLM QA Analysis Dashboard")
    
    # Initialize components
    data_loader = DataLoader()
    qa_analyzer = QAAnalyzer(data_loader)
    vehicle_analyzer = VehicleStateAnalyzer(data_loader)
    
    # Get QA distribution data
    totals = qa_analyzer.analyze_scenes()  # This gets data for all scenes
    
    # Extract totals
    total = totals["total"]
    
    # Create pie chart data
    qa_types = ['perception', 'planning', 'prediction', 'behavior']
    values = [totals[qa_type] for qa_type in qa_types]
    
    # Create pie chart
    fig = px.pie(
        values=values,
        names=[qa_type.capitalize() for qa_type in qa_types],
        title="QA Distribution Across All Scenes",
        color_discrete_sequence=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # QA Content Analysis Section
    st.header("QA Content Analysis")
    
    # Get content analysis data
    qa_content = qa_analyzer.analyze_qa_content()
    
    # Create tabs for QA analysis
    qa_tab1, qa_tab2, qa_tab3 = st.tabs(["Object Mentions", "Question-Answer Patterns", "Answer Characteristics"])
    
    with qa_tab1:
        st.subheader("Object Mentions")
        object_data = qa_content['objects']
        
        # Create bar chart for object mentions
        fig_objects = px.bar(
            x=list(object_data.keys()),
            y=list(object_data.values()),
            labels={'x': 'Objects', 'y': 'Mention Count'}
        )
        fig_objects.update_xaxes(tickangle=45)
        st.plotly_chart(fig_objects, use_container_width=True)
        
        # Object mentions by QA type
        st.subheader("Object Mentions by QA Type")
        objects_by_type = qa_content['objects_by_type']
        
        # Create grouped bar chart for object mentions by QA type
        # Prepare data for grouped bar chart
        chart_data = []
        for qa_type, objects in objects_by_type.items():
            for obj, count in objects.items():
                chart_data.append({
                    'QA Type': qa_type.capitalize(),
                    'Object': obj,
                    'Count': count
                })
        
        df_objects_by_type = pd.DataFrame(chart_data)
        
        # Create grouped bar chart
        fig_objects_by_type = px.bar(
            df_objects_by_type,
            x='Object',
            y='Count',
            color='QA Type',
            barmode='group'
        )
        fig_objects_by_type.update_xaxes(tickangle=45)
        st.plotly_chart(fig_objects_by_type, use_container_width=True)
    
    with qa_tab2:
        st.subheader("Question Patterns by QA Type")
        question_patterns = qa_content['question_patterns']
        
        # Create heatmap for question patterns
        pattern_df = pd.DataFrame(question_patterns).T
        fig_patterns = px.imshow(
            pattern_df,
            labels=dict(x="QA Type", y="Pattern", color="Count")
        )
        st.plotly_chart(fig_patterns, use_container_width=True)
        
        st.subheader("Answer Patterns by QA Type")
        answer_patterns = qa_content['answer_patterns']
        
        # Create heatmap for answer patterns
        answer_pattern_df = pd.DataFrame(answer_patterns).T
        fig_answer_patterns = px.imshow(
            answer_pattern_df,
            labels=dict(x="QA Type", y="Pattern", color="Count")
        )
        st.plotly_chart(fig_answer_patterns, use_container_width=True)
    
    with qa_tab3:
        st.subheader("Answer Characteristics")
        answer_chars = qa_content['answer_characteristics']
        
        # Create box plot for answer lengths
        length_data = []
        for qa_type, lengths in answer_chars['lengths'].items():
            for length in lengths:
                length_data.append({'QA Type': qa_type.capitalize(), 'Answer Length': length})
        
        length_df = pd.DataFrame(length_data)
        fig_lengths = px.box(
            length_df, 
            x='QA Type', 
            y='Answer Length'
        )
        st.plotly_chart(fig_lengths, use_container_width=True)
    
    # Vehicle State Analysis Section
    st.header("Vehicle State Analysis")
    
    # Get vehicle state analysis data for all scenes
    all_scenes_data = {}
    for scene_id in range(1, 7):
        try:
            scene_analysis = vehicle_analyzer.analyze_scene(scene_id)
            all_scenes_data[f"Scene {scene_id}"] = scene_analysis
        except Exception as e:
            st.error(f"Error analyzing Scene {scene_id}: {e}")
            continue
    
    # Create tabs for vehicle analysis
    vehicle_tab1, vehicle_tab2 = st.tabs(["Velocity Summary", "Driving Style"])
    
    with vehicle_tab1:
        st.subheader("Velocity Summary")
        
        # Prepare velocity data
        velocity_data = []
        for scene_name, scene_data in all_scenes_data.items():
            if scene_data and 'velocity_summary' in scene_data:
                vel_summary = scene_data['velocity_summary']
                velocity_data.append({
                    'Scene': scene_name,
                    'Avg Speed (m/s)': vel_summary.get('avg_speed', 0),
                    'Max Speed (m/s)': vel_summary.get('max_speed', 0),
                    'Avg Acceleration (m/s²)': vel_summary.get('avg_acceleration', 0),
                    'Total Distance (m)': vel_summary.get('total_distance', 0),
                    'Total Duration (s)': vel_summary.get('total_duration', 0)
                })
        
        if velocity_data:
            vel_df = pd.DataFrame(velocity_data)
            
            # Speed comparison
            fig_speed = px.bar(
                vel_df,
                x='Scene',
                y=['Avg Speed (m/s)', 'Max Speed (m/s)'],
                barmode='group'
            )
            st.plotly_chart(fig_speed, use_container_width=True)
            
            # Distance and duration
            fig_distance = px.bar(
                vel_df,
                x='Scene',
                y='Total Distance (m)'
            )
            st.plotly_chart(fig_distance, use_container_width=True)
            
            # Show detailed metrics table
            st.subheader("Detailed Velocity Metrics")
            st.dataframe(vel_df.set_index('Scene'))
    
    with vehicle_tab2:
        st.subheader("Driving Style Analysis")
        
        # Prepare driving style data
        style_data = []
        for scene_name, scene_data in all_scenes_data.items():
            if scene_data and 'driving_style' in scene_data:
                style = scene_data['driving_style']
                style_data.append({
                    'Scene': scene_name,
                    'Style': style.get('style', 'unknown'),
                    'Score': style.get('overall_score', 0),
                    'Speed Score': style.get('speed_score', 0),
                    'Acceleration Score': style.get('acceleration_score', 0),
                    'Curvature Score': style.get('curvature_score', 0)
                })
        
        if style_data:
            style_df = pd.DataFrame(style_data)
            
            # Driving style distribution
            fig_style = px.bar(
                style_df,
                x='Scene',
                y='Score',
                color='Style'
            )
            st.plotly_chart(fig_style, use_container_width=True)
            
            # Style component breakdown
            fig_components = px.bar(
                style_df,
                x='Scene',
                y=['Speed Score', 'Acceleration Score', 'Curvature Score'],
                barmode='group'
            )
            st.plotly_chart(fig_components, use_container_width=True)

if __name__ == "__main__":
    main() 