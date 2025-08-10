"""
Dashboard Generator

Creates visualizations and dashboards for the analysis results.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Any, Tuple
from pathlib import Path
from loguru import logger

from .data_loader import DataLoader
from .qa_analyzer import QAAnalyzer


class DashboardGenerator:
    """Generate visualizations and dashboards"""
    
    def __init__(self, data_loader: DataLoader, output_dir: str = "reports"):
        """
        Initialize the dashboard generator.
        
        Args:
            data_loader: DataLoader instance
            output_dir: Directory to save visualizations
        """
        self.data_loader = data_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.qa_analyzer = QAAnalyzer(data_loader)
        
        # Set style for better looking plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_question_distribution_charts(self, question_stats: Dict[str, Any] = None) -> None:
        """
        Create charts for question distribution analysis.
        
        Args:
            question_stats: Question distribution statistics (optional, will generate if None)
        """
        logger.info("Creating question distribution charts...")
        
        total_qa_distribution = self.qa_analyzer.analyze_scenes()
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('QA Distribution Analysis Across Scenes', fontsize=16, fontweight='bold')
        
        # 1. Bar chart: QA types by scene
        self._create_qa_types_bar_chart(ax1, question_stats)
        
        # 2. Pie chart: Overall QA type distribution
        self._create_overall_qa_pie_chart(ax2, question_stats)
        
        # 3. Stacked bar chart: QA distribution per scene
        self._create_stacked_qa_chart(ax3, question_stats)
        
        # 4. Line chart: QA trends across scenes
        self._create_qa_trends_chart(ax4, question_stats)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'qa_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"QA distribution charts saved to {self.output_dir / 'qa_distribution_analysis.png'}")
    
    def _create_qa_types_bar_chart(self, ax, question_stats: Dict[str, Any]) -> None:
        """Create bar chart showing QA types by scene"""
        qa_types = ['perception', 'planning', 'prediction', 'behavior']
        scenes = list(question_stats.keys())
        
        # Prepare data
        data = {qa_type: [question_stats[scene][qa_type] for scene in scenes] for qa_type in qa_types}
        
        # Create grouped bar chart
        x = range(len(scenes))
        width = 0.2
        
        for i, qa_type in enumerate(qa_types):
            ax.bar([xi + i * width for xi in x], data[qa_type], width, 
                   label=qa_type.capitalize(), alpha=0.8)
        
        ax.set_xlabel('Scenes')
        ax.set_ylabel('Number of Questions')
        ax.set_title('QA Types Distribution by Scene')
        ax.set_xticks([xi + width * 1.5 for xi in x])
        ax.set_xticklabels(scenes, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_overall_qa_pie_chart(self, ax, question_stats: Dict[str, Any]) -> None:
        """Create pie chart showing overall QA type distribution"""
        qa_types = ['perception', 'planning', 'prediction', 'behavior']
        
        # Sum up all scenes
        total_by_type = {qa_type: 0 for qa_type in qa_types}
        for scene_data in question_stats.values():
            for qa_type in qa_types:
                total_by_type[qa_type] += scene_data[qa_type]
        
        # Create pie chart
        sizes = [total_by_type[qa_type] for qa_type in qa_types]
        labels = [qa_type.capitalize() for qa_type in qa_types]
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                         startangle=90)
        ax.set_title('Overall QA Type Distribution')
        
        # Add total count in the center
        total_questions = sum(sizes)
        ax.text(0, 0, f'Total:\n{total_questions}', ha='center', va='center', 
                fontsize=12, fontweight='bold')
    
    def _create_stacked_qa_chart(self, ax, question_stats: Dict[str, Any]) -> None:
        """Create stacked bar chart showing QA distribution per scene"""
        qa_types = ['perception', 'planning', 'prediction', 'behavior']
        scenes = list(question_stats.keys())
        
        # Prepare data for stacked bar
        bottom = [0] * len(scenes)
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        for i, qa_type in enumerate(qa_types):
            values = [question_stats[scene][qa_type] for scene in scenes]
            ax.bar(scenes, values, bottom=bottom, label=qa_type.capitalize(), 
                   color=colors[i], alpha=0.8)
            bottom = [b + v for b, v in zip(bottom, values)]
        
        ax.set_xlabel('Scenes')
        ax.set_ylabel('Number of Questions')
        ax.set_title('Stacked QA Distribution by Scene')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _create_qa_trends_chart(self, ax, question_stats: Dict[str, Any]) -> None:
        """Create line chart showing QA trends across scenes"""
        qa_types = ['perception', 'planning', 'prediction', 'behavior']
        scenes = list(question_stats.keys())
        
        # Prepare data
        for qa_type in qa_types:
            values = [question_stats[scene][qa_type] for scene in scenes]
            ax.plot(scenes, values, marker='o', linewidth=2, markersize=6, 
                   label=qa_type.capitalize())
        
        ax.set_xlabel('Scenes')
        ax.set_ylabel('Number of Questions')
        ax.set_title('QA Trends Across Scenes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def create_object_distribution_charts(self, object_stats: Dict[str, Any]) -> None:
        """
        Create charts for object distribution analysis.
        
        Args:
            object_stats: Object distribution statistics
        """
        # TODO: Implement object distribution visualizations
        pass
    
    def create_scenario_analysis_charts(self, scenario_stats: Dict[str, Any]) -> None:
        """
        Create charts for scenario analysis.
        
        Args:
            scenario_stats: Scenario analysis statistics
        """
        # TODO: Implement scenario analysis visualizations
        pass
    
    def create_pattern_analysis_charts(self, pattern_stats: Dict[str, Any]) -> None:
        """
        Create charts for pattern analysis.
        
        Args:
            pattern_stats: Pattern analysis statistics
        """
        # TODO: Implement pattern analysis visualizations
        pass
    
    def create_sample_question_visualizations(self, sample_questions: List[Dict[str, Any]]) -> None:
        """
        Create visualizations for interesting sample questions.
        
        Args:
            sample_questions: List of interesting sample questions
        """
        # TODO: Implement sample question visualizations
        pass
    
    def generate_dashboard_report(self, all_stats: Dict[str, Any]) -> str:
        """
        Generate comprehensive dashboard report.
        
        Args:
            all_stats: All analysis statistics
            
        Returns:
            Path to generated dashboard
        """
        # TODO: Implement comprehensive dashboard generation
        pass
    
    def create_interactive_dashboard(self, all_stats: Dict[str, Any]) -> str:
        """
        Create interactive dashboard using plotly or similar.
        
        Args:
            all_stats: All analysis statistics
            
        Returns:
            Path to interactive dashboard
        """
        # TODO: Implement interactive dashboard
        pass 