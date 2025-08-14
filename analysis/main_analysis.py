"""
Main Analysis Orchestrator

Runs all analysis components and generates deliverables.
"""

from typing import Dict, Any
from loguru import logger

from parsers.data_service import DataService
from .vehicle_state_analyzer import VehicleStateAnalyzer
from .sensor_analyzer import SensorAnalyzer
from .predictor_analyzer import PredictorAnalyzer
from .qa_analyzer import QAAnalyzer
from .dashboard_generator import DashboardGenerator
from .result import AnalysisResult, ResultHandler
from config import get_config


class MainAnalysis:
    """Main analysis orchestrator"""
    
    def __init__(self, data_path: str = None):
        """
        Initialize the main analysis.
        
        Args:
            data_path: Path to the concatenated data file (optional, uses config if not provided)
        """
        # Get configuration
        self.config = get_config()
        
        # Use provided data path or get from config
        if data_path is None:
            data_path = self.config.get_data_path()
        
        # Initialize data service
        self.data_service = DataService(data_path)
        
        # Initialize analyzers
        self.vehicle_analyzer = VehicleStateAnalyzer(self.data_service)
        self.sensor_analyzer = SensorAnalyzer(self.data_service)
        self.predictor_analyzer = PredictorAnalyzer(self.data_service)
        self.qa_analyzer = QAAnalyzer(self.data_service)
        self.dashboard_generator = DashboardGenerator(self.data_service)
    
    def run_vehicle_analysis(self) -> AnalysisResult:
        """
        Run vehicle state analysis.
        
        Returns:
            Vehicle state analysis results
        """
        logger.info("Running vehicle state analysis...")
        
        try:
            # Analyze all scenes
            results = self.vehicle_analyzer.analyze_all_scenes()
            
            return AnalysisResult.success(
                data=results,
                metadata={'analyzer': 'vehicle_state', 'scenes_analyzed': len(results)}
            )
            
        except Exception as e:
            logger.error(f"Vehicle analysis failed: {e}")
            return AnalysisResult.error(str(e), metadata={'analyzer': 'vehicle_state'})
    
    def run_question_analysis(self) -> AnalysisResult:
        """
        Run question distribution analysis.
        
        Returns:
            Question analysis results
        """
        logger.info("Running question distribution analysis...")
        
        try:
            # Analyze all scenes
            results = self.qa_analyzer.analyze_all_scenes()
            
            return AnalysisResult.success(
                data=results,
                metadata={'analyzer': 'qa', 'scenes_analyzed': len(results)}
            )
            
        except Exception as e:
            logger.error(f"Question analysis failed: {e}")
            return AnalysisResult.error(str(e), metadata={'analyzer': 'qa'})
    
    def run_sensor_analysis(self) -> AnalysisResult:
        """
        Run sensor data analysis.
        
        Returns:
            Sensor analysis results
        """
        logger.info("Running sensor data analysis...")
        
        try:
            # Analyze all scenes
            results = self.sensor_analyzer.analyze_all_scenes()
            
            return AnalysisResult.success(
                data=results,
                metadata={'analyzer': 'sensor', 'scenes_analyzed': len(results)}
            )
            
        except Exception as e:
            logger.error(f"Sensor analysis failed: {e}")
            return AnalysisResult.error(str(e), metadata={'analyzer': 'sensor'})
    
    def run_predictor_analysis(self) -> AnalysisResult:
        """
        Run predictor analysis.
        
        Returns:
            Predictor analysis results
        """
        logger.info("Running predictor analysis...")
        
        try:
            # Analyze all scenes
            results = self.predictor_analyzer.analyze_all_scenes()
            
            return AnalysisResult.success(
                data=results,
                metadata={'analyzer': 'predictor', 'scenes_analyzed': len(results)}
            )
            
        except Exception as e:
            logger.error(f"Predictor analysis failed: {e}")
            return AnalysisResult.error(str(e), metadata={'analyzer': 'predictor'})
    
    def run_pattern_analysis(self) -> AnalysisResult:
        """
        Run pattern and anomaly analysis.
        
        Returns:
            Pattern analysis results
        """
        logger.info("Running pattern analysis...")
        
        try:
            # This could combine results from multiple analyzers
            # For now, return a placeholder
            return AnalysisResult.success(
                data={'message': 'Pattern analysis not yet implemented'},
                metadata={'analyzer': 'pattern'}
            )
            
        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
            return AnalysisResult.error(str(e), metadata={'analyzer': 'pattern'})
    
    def generate_dashboard(self, all_results: Dict[str, Any]) -> str:
        """
        Generate comprehensive dashboard.
        
        Args:
            all_results: All analysis results
            
        Returns:
            Path to generated dashboard
        """
        logger.info("Generating dashboard...")
        # TODO: Implement dashboard generation
        return "reports/dashboard.html"
    
    def generate_markdown_report(self, all_results: Dict[str, Any]) -> str:
        """
        Generate markdown report with findings.
        
        Args:
            all_results: All analysis results
            
        Returns:
            Path to generated report
        """
        logger.info("Generating markdown report...")
        # TODO: Implement markdown report generation
        return "reports/analysis_report.md"
    
    def run_complete_analysis(self) -> AnalysisResult:
        """
        Run complete analysis pipeline.
        
        Returns:
            Analysis result with all analysis results and report paths
        """
        logger.info("Starting complete analysis pipeline...")
        
        # Run all analysis components
        vehicle_results = self.run_vehicle_analysis()
        question_results = self.run_question_analysis()
        sensor_results = self.run_sensor_analysis()
        predictor_results = self.run_predictor_analysis()
        pattern_results = self.run_pattern_analysis()
        
        # Combine all results
        all_results = {
            'vehicle_analysis': vehicle_results,
            'question_analysis': question_results,
            'sensor_analysis': sensor_results,
            'predictor_analysis': predictor_results,
            'pattern_analysis': pattern_results
        }
        
        # Check if any analysis failed
        failed_analyses = []
        successful_analyses = {}
        
        for name, result in all_results.items():
            if result.status.value == 'success':
                successful_analyses[name] = result.data
            else:
                failed_analyses.append(f"{name}: {result.error}")
        
        # Generate deliverables if we have successful analyses
        dashboard_path = None
        report_path = None
        
        if successful_analyses:
            try:
                dashboard_path = self.generate_dashboard(successful_analyses)
                report_path = self.generate_markdown_report(successful_analyses)
            except Exception as e:
                logger.error(f"Failed to generate deliverables: {e}")
        
        # Create final result
        if failed_analyses:
            warning = f"Some analyses failed: {'; '.join(failed_analyses)}"
            return AnalysisResult.partial(
                data={
                    'analysis_results': successful_analyses,
                    'dashboard_path': dashboard_path,
                    'report_path': report_path
                },
                warning=warning,
                metadata={'failed_analyses': failed_analyses}
            )
        else:
            return AnalysisResult.success(
                data={
                    'analysis_results': successful_analyses,
                    'dashboard_path': dashboard_path,
                    'report_path': report_path
                },
                metadata={'total_analyses': len(all_results)}
            )


def main():
    """Main entry point for analysis"""
    analyzer = MainAnalysis()
    result = analyzer.run_complete_analysis()
    
    if result.status.value == 'success':
        logger.info(f"Analysis completed successfully!")
        logger.info(f"Dashboard: {result.data.get('dashboard_path', 'Not generated')}")
        logger.info(f"Report: {result.data.get('report_path', 'Not generated')}")
    elif result.status.value == 'partial':
        logger.warning(f"Analysis completed with warnings: {result.warning}")
        logger.info(f"Dashboard: {result.data.get('dashboard_path', 'Not generated')}")
        logger.info(f"Report: {result.data.get('report_path', 'Not generated')}")
    else:
        logger.error(f"Analysis failed: {result.error}")


if __name__ == "__main__":
    main() 