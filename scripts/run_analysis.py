#!/usr/bin/env python3
"""
Main Analysis Runner Script

This script serves as the entry point for running the complete DriveLM data analysis pipeline.
It can be run in different modes: full analysis, dashboard only, or specific analysis components.

USAGE EXAMPLES:
    # Run full analysis locally (generates static reports)
    python scripts/run_analysis.py --mode full
    
    # Run interactive dashboard locally on port 8501
    python scripts/run_analysis.py --mode local --port 8501
    
    # Run dashboard only locally
    python scripts/run_analysis.py --mode dashboard --port 8501
    
    # Run full analysis in Docker
    python scripts/run_analysis.py --mode docker --docker-service drivelm-analysis
    
    # Run dashboard only in Docker
    python scripts/run_analysis.py --mode docker --docker-service drivelm-dashboard --port 8502
    
    # Run analysis without dashboard in Docker
    python scripts/run_analysis.py --mode docker --docker-service drivelm-analysis-only

DASHBOARD ACCESS:
    - Local: http://localhost:8501
    - Docker: http://localhost:8501 (or specified port)
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
    logger.info(f"Starting interactive Streamlit dashboard on port {port}...")
    logger.info(f"Access the dashboard at: http://localhost:{port}")
    
    # Set environment variable for data path
    os.environ['DATA_PATH'] = data_path
    
    # Run streamlit dashboard
    import subprocess
    cmd = [
        "streamlit", "run", 
        str(project_root / "analysis" / "dashboard.py"),
        "--server.port", str(port),
        "--server.address", "0.0.0.0",
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false"
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


def run_docker_analysis(service_name: str, port: int = 8501):
    """
    Run analysis using Docker Compose
    
    Args:
        service_name: Name of the Docker service to run
        port: Port for the dashboard (if applicable)
    """
    logger.info(f"Starting Docker service: {service_name}")
    logger.info(f"Dashboard will be available at: http://localhost:{port}")
    
    import subprocess
    
    # Check if docker-compose.yml exists
    compose_file = project_root / "docker-compose.yml"
    if not compose_file.exists():
        logger.error("docker-compose.yml not found in project root")
        sys.exit(1)
    
    # Build and run the service
    try:
        # Build the service
        logger.info("Building Docker service...")
        subprocess.run([
            "docker-compose", "build", service_name
        ], check=True, cwd=project_root)
        
        # Run the service
        logger.info(f"Starting {service_name}...")
        subprocess.run([
            "docker-compose", "up", service_name
        ], check=True, cwd=project_root)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Docker command failed: {e}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error("Docker or docker-compose not found. Please install Docker and Docker Compose.")
        sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="DriveLM Data Analysis Pipeline")
    parser.add_argument(
        "--mode", 
        choices=["full", "dashboard", "analysis", "local", "docker"],
        default="full",
        help="Analysis mode to run (local/docker for execution environment)"
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
    parser.add_argument(
        "--docker-service",
        type=str,
        choices=["drivelm-analysis", "drivelm-dashboard", "drivelm-analysis-only"],
        default="drivelm-analysis",
        help="Docker service to use (docker mode only)"
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
            
        elif args.mode == "local":
            # Run locally with interactive dashboard
            logger.info("Running analysis locally with interactive dashboard...")
            run_dashboard_only(data_path, args.port)
            
        elif args.mode == "docker":
            # Run using Docker
            run_docker_analysis(args.docker_service, args.port)
            
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()