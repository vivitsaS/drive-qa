#!/bin/bash

# DriveLM Analysis Docker Runner Script
# Usage: ./docker/run.sh [mode] [options]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
MODE="full"
PORT=8501
DATA_PATH="./data"
REPORTS_PATH="./reports"
LOGS_PATH="./logs"

# Function to print usage
usage() {
    echo "Usage: $0 [mode] [options]"
    echo ""
    echo "Modes:"
    echo "  full        Run complete analysis pipeline (default)"
    echo "  dashboard   Run interactive dashboard only"
    echo "  analysis    Run analysis without dashboard"
    echo "  build       Build Docker image"
    echo "  clean       Clean up containers and images"
    echo ""
    echo "Options:"
    echo "  --port PORT        Port for dashboard (default: 8501)"
    echo "  --data PATH        Data directory path (default: ./data)"
    echo "  --reports PATH     Reports output path (default: ./reports)"
    echo "  --logs PATH        Logs output path (default: ./logs)"
    echo "  --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build                           # Build the image"
    echo "  $0 full                           # Run full analysis"
    echo "  $0 dashboard --port 8502          # Run dashboard on port 8502"
    echo "  $0 analysis --data /path/to/data  # Run analysis with custom data path"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}Error: Docker is not running${NC}"
        exit 1
    fi
}

# Function to create necessary directories
create_dirs() {
    mkdir -p "$DATA_PATH" "$REPORTS_PATH" "$LOGS_PATH"
    echo -e "${GREEN}Created directories: $DATA_PATH, $REPORTS_PATH, $LOGS_PATH${NC}"
}

# Function to build Docker image
build_image() {
    echo -e "${YELLOW}Building DriveLM Analysis Docker image...${NC}"
    docker build -f docker/Dockerfile -t drivelm-analysis .
    echo -e "${GREEN}Docker image built successfully${NC}"
}

# Function to run analysis
run_analysis() {
    local mode=$1
    
    echo -e "${YELLOW}Running DriveLM Analysis in $mode mode...${NC}"
    
    # Base docker run command
    local cmd="docker run --rm"
    
    # Add port mapping for dashboard mode
    if [ "$mode" = "dashboard" ] || [ "$mode" = "full" ]; then
        cmd="$cmd -p $PORT:8501"
    fi
    
    # Add volume mounts
    cmd="$cmd -v $(realpath $DATA_PATH):/app/data"
    cmd="$cmd -v $(realpath $REPORTS_PATH):/app/reports"
    cmd="$cmd -v $(realpath $LOGS_PATH):/app/logs"
    
    # Add environment variables
    cmd="$cmd -e PYTHONPATH=/app"
    cmd="$cmd -e DATA_PATH=/app/data/concatenated_data/concatenated_data.json"
    
    # Add image name and mode
    cmd="$cmd drivelm-analysis $mode"
    
    echo -e "${YELLOW}Executing: $cmd${NC}"
    eval $cmd
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Analysis completed successfully${NC}"
        if [ "$mode" = "dashboard" ]; then
            echo -e "${GREEN}Dashboard is running at: http://localhost:$PORT${NC}"
        fi
    else
        echo -e "${RED}Analysis failed${NC}"
        exit 1
    fi
}

# Function to clean up Docker resources
clean_up() {
    echo -e "${YELLOW}Cleaning up Docker resources...${NC}"
    
    # Stop and remove containers
    docker ps -a --filter "ancestor=drivelm-analysis" --format "{{.ID}}" | xargs -r docker rm -f
    
    # Remove image
    docker rmi -f drivelm-analysis 2>/dev/null || true
    
    echo -e "${GREEN}Cleanup completed${NC}"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        full|dashboard|analysis|build|clean)
            MODE="$1"
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --data)
            DATA_PATH="$2"
            shift 2
            ;;
        --reports)
            REPORTS_PATH="$2"
            shift 2
            ;;
        --logs)
            LOGS_PATH="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Main execution
check_docker

case $MODE in
    build)
        build_image
        ;;
    clean)
        clean_up
        ;;
    full|dashboard|analysis)
        create_dirs
        # Check if image exists, build if not
        if ! docker image inspect drivelm-analysis > /dev/null 2>&1; then
            echo -e "${YELLOW}Docker image not found, building...${NC}"
            build_image
        fi
        run_analysis $MODE
        ;;
    *)
        echo -e "${RED}Invalid mode: $MODE${NC}"
        usage
        exit 1
        ;;
esac