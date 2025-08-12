# DriveLM Data Analysis - Docker Setup

This document provides instructions for running the DriveLM data analysis pipeline using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose (usually included with Docker)
- DriveLM dataset and NuScenes-mini data

## Quick Start

### 1. Build the Docker Image

```bash
# Build the image using docker-compose
docker-compose build drivelm-analysis

# Or build directly with Docker
docker build -f docker/Dockerfile -t drivelm-analysis .
```

### 2. Prepare Your Data

- download v1_1_train_nus.json from drivelm
- download nuscenes mini data set from the nuscenes website
- keep them both in data/
- there are only 6 overlapping scenes between the 2.
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

### 3. Run Analysis

Choose one of the following modes:

#### Full Analysis Pipeline
Runs complete analysis and generates reports + dashboard:
```bash
docker-compose up drivelm-analysis
```

#### Dashboard Only
Runs interactive Streamlit dashboard:
```bash
docker-compose up drivelm-dashboard
```
Access the dashboard at: http://localhost:8501

#### Analysis Only
Runs analysis without interactive dashboard:
```bash
docker-compose up drivelm-analysis-only
```

## Advanced Usage

### Custom Data Path

If your data is in a different location:
```bash
docker run -v /path/to/your/data:/app/data \
           -v ./reports:/app/reports \
           drivelm-analysis
```

### Run Specific Analysis Mode

```bash
# Full analysis
docker run drivelm-analysis full

# Dashboard only
docker run -p 8501:8501 drivelm-analysis dashboard

# Analysis only
docker run drivelm-analysis analysis
```

### Development Mode

For development with live code changes:
```bash
docker run -v $(pwd):/app \
           -v ./data:/app/data \
           -v ./reports:/app/reports \
           -p 8501:8501 \
           drivelm-analysis dashboard
```

## Output Files

After running the analysis, you'll find:

- **Reports**: `./reports/` directory
  - Analysis markdown reports
  - Generated visualizations
  - Dashboard HTML files

- **Logs**: `./logs/` directory
  - Analysis execution logs
  - Error logs and debugging information

## Configuration

### Environment Variables

- `DATA_PATH`: Path to your data file or directory
- `PYTHONPATH`: Python path (set to `/app` by default)
- `STREAMLIT_SERVER_*`: Streamlit configuration for dashboard mode

### Volume Mounts

The docker-compose setup automatically mounts:
- `./data` → `/app/data` (your dataset)
- `./reports` → `/app/reports` (analysis outputs)
- `./logs` → `/app/logs` (execution logs)

## Troubleshooting

### Common Issues

1. **Permission Errors**
   ```bash
   # Fix file permissions
   sudo chown -R $USER:$USER ./data ./reports ./logs
   ```

2. **Port Already in Use**
   ```bash
   # Use different port
   docker run -p 8502:8501 drivelm-analysis dashboard
   ```

3. **Memory Issues**
   ```bash
   # Increase Docker memory limit in Docker Desktop settings
   # Or run with memory limit
   docker run --memory=4g drivelm-analysis
   ```

4. **Data Not Found**
   - Ensure data path is correctly mounted
   - Check that `concatenated_data.json` exists in the expected location
   - Verify file permissions

### Debug Mode

Run with verbose logging:
```bash
docker run -e LOG_LEVEL=DEBUG drivelm-analysis
```

### Check Container Logs

```bash
# View logs for running container
docker logs drivelm-analysis

# Follow logs in real-time
docker logs -f drivelm-analysis
```

## Performance Optimization

### Resource Allocation

For large datasets, allocate more resources:
```bash
docker run --cpus=4 --memory=8g drivelm-analysis
```

### Parallel Processing

The analysis pipeline supports parallel processing. Adjust in your analysis configuration files.

## Integration with CI/CD

### Example GitHub Actions

```yaml
name: DriveLM Analysis
on: [push]
jobs:
  analysis:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Analysis
        run: |
          docker build -f docker/Dockerfile -t drivelm-analysis .
          docker run -v ./data:/app/data drivelm-analysis analysis
      - name: Upload Reports
        uses: actions/upload-artifact@v2
        with:
          name: analysis-reports
          path: reports/
```

## Support

For issues related to:
- **Docker setup**: Check this documentation and Docker logs
- **Analysis pipeline**: Check analysis logs in `./logs/`
- **Data loading**: Verify data structure and file permissions

## Next Steps

After successful dockerization:
1. Set up automated analysis runs
2. Configure continuous integration
3. Deploy dashboard to cloud platforms
4. Set up monitoring and alerting