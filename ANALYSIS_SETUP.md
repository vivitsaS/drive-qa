# DriveLM Data Analysis - Docker Setup

This document provides instructions for running the DriveLM data analysis pipeline using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose (usually included with Docker)
- DriveLM dataset and NuScenes-mini data

## Quick Start



### 1. Prepare Your Data

- download v1_1_train_nus.json from drivelm
- download nuscenes mini data set from the nuscenes website
- keep them both in data/
- run the parsers/find_overlapping_scenes.py script to get a csv of overlapping scenes(there are only 6 overlapping scenes between the 2)
- run parsers/concatenate.py to concatenate the 2 sources.

Ensure your data is organized in the following structure:
```
data/
├── concatenated_data
│   ├── concatenated_data.json
│   ├── first_entry.json
│   └── overlapping_scenes.csv
├── v1_1_train_nus.json
└── v1.0-mini
    ├── ..
```
### 2. Build the Docker Image

```bash
# Build the image using docker-compose
docker-compose build drivelm-analysis

# Or build directly with Docker
docker build -f docker/Dockerfile -t drivelm-analysis .
```

### 3. Run Analysis

Choose one of the following modes:

#### Run Dashboard
Runs interactive Streamlit dashboard:
```bash
docker compose up drivelm-dashboard
```
Access the dashboard at: http://localhost:8501
