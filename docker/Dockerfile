FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/concatenated_data reports logs

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Expose port for Streamlit dashboard
EXPOSE 8501

# Create entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$1" = "dashboard" ]; then\n\
    exec python scripts/run_analysis.py --mode dashboard --port 8501\n\
elif [ "$1" = "analysis" ]; then\n\
    exec python scripts/run_analysis.py --mode analysis\n\
else\n\
    exec python scripts/run_analysis.py --mode full\n\
fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Default command
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["full"] 