"""
Unified Visualizer

Provides a unified interface for creating visualizations from analysis results.
Works with the standardized AnalysisResult format.
"""

from typing import Dict, Any, List, Optional, Union
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from loguru import logger

from analysis.result import AnalysisResult, ResultStatus


class UnifiedVisualizer:
    """Unified visualization interface for analysis results"""
    
    def __init__(self):
        """Initialize the unified visualizer"""
        self.default_colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#99ccff']
    
    def create_qa_distribution_chart(self, qa_results: AnalysisResult) -> Optional[go.Figure]:
        """
        Create QA distribution pie chart.
        
        Args:
            qa_results: AnalysisResult containing QA analysis data
            
        Returns:
            Plotly figure or None if data is not available
        """
        try:
            if not qa_results.status.value == 'success':
                logger.warning("QA results not successful, cannot create chart")
                return None
            
            data = qa_results.data
            
            # Extract QA distribution data
            if 'total' in data:
                totals = data['total']
            elif 'summary' in data and 'qa_distribution' in data['summary']:
                totals = data['summary']['qa_distribution']
            else:
                logger.warning("No QA distribution data found")
                return None
            
            qa_types = ['perception', 'planning', 'prediction', 'behavior']
            values = [totals.get(qa_type, 0) for qa_type in qa_types]
            
            # Create pie chart
            fig = px.pie(
                values=values,
                names=[qa_type.capitalize() for qa_type in qa_types],
                title="QA Distribution Across All Scenes",
                color_discrete_sequence=self.default_colors
            )
            
            fig.update_layout(
                title_x=0.5,
                showlegend=True,
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating QA distribution chart: {e}")
            return None
    
    def create_vehicle_analysis_chart(self, vehicle_results: AnalysisResult) -> Optional[go.Figure]:
        """
        Create vehicle analysis charts.
        
        Args:
            vehicle_results: AnalysisResult containing vehicle analysis data
            
        Returns:
            Plotly figure or None if data is not available
        """
        try:
            if not vehicle_results.status.value == 'success':
                logger.warning("Vehicle results not successful, cannot create chart")
                return None
            
            data = vehicle_results.data
            
            # Extract scene data
            scenes_data = []
            for scene_key, scene_data in data.items():
                if scene_key.startswith('scene_'):
                    scene_id = scene_key.replace('scene_', '')
                    if 'velocity_summary' in scene_data:
                        vel_data = scene_data['velocity_summary']
                        scenes_data.append({
                            'scene_id': scene_id,
                            'avg_speed': vel_data.get('avg_speed', 0),
                            'max_speed': vel_data.get('max_speed', 0),
                            'avg_acceleration': vel_data.get('avg_acceleration', 0),
                            'total_distance': vel_data.get('total_distance', 0)
                        })
            
            if not scenes_data:
                logger.warning("No vehicle data found for visualization")
                return None
            
            df = pd.DataFrame(scenes_data)
            
            # Create subplots
            fig = go.Figure()
            
            # Add speed metrics
            fig.add_trace(go.Bar(
                x=df['scene_id'],
                y=df['avg_speed'],
                name='Average Speed',
                marker_color='#66b3ff'
            ))
            
            fig.add_trace(go.Bar(
                x=df['scene_id'],
                y=df['max_speed'],
                name='Max Speed',
                marker_color='#ff9999'
            ))
            
            fig.update_layout(
                title="Vehicle Speed Analysis by Scene",
                xaxis_title="Scene ID",
                yaxis_title="Speed (m/s)",
                barmode='group',
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating vehicle analysis chart: {e}")
            return None
    
    def create_sensor_analysis_chart(self, sensor_results: AnalysisResult) -> Optional[go.Figure]:
        """
        Create sensor analysis charts.
        
        Args:
            sensor_results: AnalysisResult containing sensor analysis data
            
        Returns:
            Plotly figure or None if data is not available
        """
        try:
            if not sensor_results.status.value == 'success':
                logger.warning("Sensor results not successful, cannot create chart")
                return None
            
            data = sensor_results.data
            
            # Extract camera activity data
            camera_data = []
            for scene_key, scene_data in data.items():
                if scene_key.startswith('scene_'):
                    scene_id = scene_key.replace('scene_', '')
                    if 'camera_activity' in scene_data:
                        activity = scene_data['camera_activity']
                        for camera, info in activity.items():
                            camera_data.append({
                                'scene_id': scene_id,
                                'camera': camera,
                                'active_samples': info.get('active_samples', 0),
                                'total_samples': info.get('total_samples', 0)
                            })
            
            if not camera_data:
                logger.warning("No sensor data found for visualization")
                return None
            
            df = pd.DataFrame(camera_data)
            
            # Create heatmap
            pivot_df = df.pivot(index='camera', columns='scene_id', values='active_samples')
            
            fig = px.imshow(
                pivot_df,
                title="Camera Activity Heatmap",
                labels=dict(x="Scene ID", y="Camera", color="Active Samples"),
                color_continuous_scale='Blues'
            )
            
            fig.update_layout(
                title_x=0.5,
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating sensor analysis chart: {e}")
            return None
    
    def create_predictor_analysis_chart(self, predictor_results: AnalysisResult) -> Optional[go.Figure]:
        """
        Create predictor analysis charts.
        
        Args:
            predictor_results: AnalysisResult containing predictor analysis data
            
        Returns:
            Plotly figure or None if data is not available
        """
        try:
            if not predictor_results.status.value == 'success':
                logger.warning("Predictor results not successful, cannot create chart")
                return None
            
            data = predictor_results.data
            
            # Extract cross-scene analysis
            if 'cross_scene_analysis' not in data:
                logger.warning("No cross-scene analysis data found")
                return None
            
            cross_analysis = data['cross_scene_analysis']
            consistent_predictors = cross_analysis.get('consistent_predictors', {})
            
            # Prepare data for visualization
            predictor_data = []
            for qa_type, predictors in consistent_predictors.items():
                for predictor, count in predictors.items():
                    predictor_data.append({
                        'qa_type': qa_type.capitalize(),
                        'predictor': predictor,
                        'consistency_count': count
                    })
            
            if not predictor_data:
                logger.warning("No consistent predictors found for visualization")
                return None
            
            df = pd.DataFrame(predictor_data)
            
            # Create bar chart
            fig = px.bar(
                df,
                x='predictor',
                y='consistency_count',
                color='qa_type',
                title="Consistent Predictors Across Scenes",
                labels={'predictor': 'Predictor Feature', 'consistency_count': 'Number of Scenes'},
                color_discrete_sequence=self.default_colors
            )
            
            fig.update_layout(
                title_x=0.5,
                xaxis_tickangle=-45,
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating predictor analysis chart: {e}")
            return None
    
    def create_summary_dashboard(self, all_results: Dict[str, AnalysisResult]) -> List[go.Figure]:
        """
        Create a summary dashboard with all available charts.
        
        Args:
            all_results: Dictionary of analysis results
            
        Returns:
            List of Plotly figures
        """
        figures = []
        
        # Create QA distribution chart
        if 'question_analysis' in all_results:
            qa_fig = self.create_qa_distribution_chart(all_results['question_analysis'])
            if qa_fig:
                figures.append(qa_fig)
        
        # Create vehicle analysis chart
        if 'vehicle_analysis' in all_results:
            vehicle_fig = self.create_vehicle_analysis_chart(all_results['vehicle_analysis'])
            if vehicle_fig:
                figures.append(vehicle_fig)
        
        # Create sensor analysis chart
        if 'sensor_analysis' in all_results:
            sensor_fig = self.create_sensor_analysis_chart(all_results['sensor_analysis'])
            if sensor_fig:
                figures.append(sensor_fig)
        
        # Create predictor analysis chart
        if 'predictor_analysis' in all_results:
            predictor_fig = self.create_predictor_analysis_chart(all_results['predictor_analysis'])
            if predictor_fig:
                figures.append(predictor_fig)
        
        return figures
    
    def create_error_summary(self, all_results: Dict[str, AnalysisResult]) -> Optional[go.Figure]:
        """
        Create a summary of errors and warnings from analysis results.
        
        Args:
            all_results: Dictionary of analysis results
            
        Returns:
            Plotly figure or None if no errors
        """
        try:
            error_data = []
            
            for analysis_name, result in all_results.items():
                if result.status.value == 'error':
                    error_data.append({
                        'analysis': analysis_name,
                        'status': 'Error',
                        'message': result.error or 'Unknown error'
                    })
                elif result.status.value == 'partial':
                    error_data.append({
                        'analysis': analysis_name,
                        'status': 'Partial Success',
                        'message': result.warning or 'Partial success with warnings'
                    })
            
            if not error_data:
                return None
            
            df = pd.DataFrame(error_data)
            
            fig = px.bar(
                df,
                x='analysis',
                color='status',
                title="Analysis Status Summary",
                color_discrete_map={
                    'Error': '#ff9999',
                    'Partial Success': '#ffcc99'
                }
            )
            
            fig.update_layout(
                title_x=0.5,
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating error summary: {e}")
            return None 