#!/usr/bin/env python3
"""
Main Analysis Runner Script

This script serves as the entry point for running the complete DriveLM data analysis pipeline.
It can be run in different modes: full analysis, dashboard only, or specific analysis components.
"""

import argparse
import os
import sys
from pathlib import Path
from loguru import logger

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analysis.main_analysis import MainAnalysis
from parsers.data_loader import DataLoader


def setup_logging():
    """Configure logging for the analysis pipeline"""
    logger.add("logs/analysis.log", rotation="10 MB", level="INFO")
    logger.info("Starting DriveLM Analysis Pipeline")


def run_full_analysis(data_path: str):
    """
    Run the complete analysis pipeline
    
    Args:
        data_path: Path to the data directory or concatenated data file
    """
    logger.info("Running full analysis pipeline...")
    
    # Initialize main analysis
    analyzer = MainAnalysis(data_path)
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    logger.info(f"Analysis complete!")
    logger.info(f"Dashboard: {results.get('dashboard_path', 'Not generated')}")
    logger.info(f"Report: {results.get('report_path', 'Not generated')}")
    
    return results


def run_dashboard_only(data_path: str, port: int = 8501):
    """
    Run only the Streamlit dashboard
    
    Args:
        data_path: Path to the data directory or concatenated data file
        port: Port to run the dashboard on
    """
    logger.info(f"Starting dashboard on port {port}...")
    
    # Set environment variable for data path
    os.environ['DATA_PATH'] = data_path
    
    # Run streamlit dashboard
    import subprocess
    cmd = [
        "streamlit", "run", 
        str(project_root / "analysis" / "dashboard.py"),
        "--server.port", str(port),
        "--server.address", "0.0.0.0"
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)


def validate_data_path(data_path: str) -> str:
    """
    Validate and return the correct data path
    
    Args:
        data_path: Input data path
        
    Returns:
        Validated data path
    """
    if not data_path:
        # Try default paths
        default_paths = [
            "data/concatenated_data/concatenated_data.json",
            "data/raw",
            "data"
        ]
        
        for path in default_paths:
            if Path(path).exists():
                logger.info(f"Using default data path: {path}")
                return path
        
        logger.error("No data path provided and no default data found")
        sys.exit(1)
    
    if not Path(data_path).exists():
        logger.error(f"Data path does not exist: {data_path}")
        sys.exit(1)
    
    return data_path


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="DriveLM Data Analysis Pipeline")
    parser.add_argument(
        "--mode", 
        choices=["full", "dashboard", "analysis"],
        default="full",
        help="Analysis mode to run"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to data directory or concatenated data file"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for dashboard (dashboard mode only)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Output directory for reports and visualizations"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Validate data path
    data_path = validate_data_path(args.data_path)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        if args.mode == "full":
            # Run complete analysis pipeline
            results = run_full_analysis(data_path)
            logger.info("Full analysis completed successfully")
            
        elif args.mode == "dashboard":
            # Run dashboard only
            run_dashboard_only(data_path, args.port)
            
        elif args.mode == "analysis":
            # Run analysis without dashboard
            analyzer = MainAnalysis(data_path)
            results = analyzer.run_complete_analysis()
            logger.info("Analysis completed successfully")
            
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()