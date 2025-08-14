# DriveLM RAG Agent

A Retrieval-Augmented Generation (RAG) agent for answering questions about autonomous driving scenes from the DriveLM dataset. This project combines multimodal AI capabilities with comprehensive driving scene analysis to provide intelligent responses to questions about perception, planning, prediction, and behavior in autonomous driving scenarios.

## 🚗 Project Overview

The DriveLM RAG Agent analyzes driving scenes by:
- Processing annotated images from 6 camera views (front, front-left, front-right, back, back-left, back-right)
- Integrating vehicle movement data (speed, acceleration, position)
- Analyzing sensor data (object detections, LiDAR/radar points)
- Providing contextual answers to driving-related questions

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   DriveLM       │    │   Context       │    │   Gemini 1.5    │
│   Dataset       │───▶│   Retriever     │───▶│   Flash Model   │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Scene Data    │    │   Vehicle &     │    │   Intelligent   │
│   (Images, QA)  │    │   Sensor Data   │    │   Responses     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
drive-qa/
├── analysis/                    # Data analysis and visualization
│   ├── base_analyzer.py        # Base analysis functionality
│   ├── qa_analyzer.py          # QA pair analysis
│   ├── predictor_analyzer.py   # Feature prediction analysis
│   ├── sensor_analyzer.py      # Sensor data analysis
│   ├── vehicle_state_analyzer.py # Vehicle movement analysis
│   └── dashboard.py            # Interactive dashboard
├── parsers/                    # Data loading and processing
│   ├── data_loader.py          # Main data loading functionality
│   ├── concatenate.py          # Data concatenation utilities
│   └── utils.py                # Utility functions
├── rag/                        # RAG agent implementation
│   ├── rag_agent.py            # Main RAG agent class
│   ├── retrieval/              # Context retrieval system
│   │   └── context_retriever.py # Scene context retrieval
│   └── MODEL_SELECTION.MD      # Model selection analysis
├── scripts/                    # Execution scripts
│   ├── run_rag_agent.py        # Single RAG agent test
│   ├── run_eval.py             # Comprehensive evaluation
│   └── run_analysis.py         # Data analysis runner
├── src/                        # Source code
│   └── visualizers/            # Data visualization tools
├── tests/                      # Unit tests
├── data/                       # Dataset storage
├── logs/                       # Execution logs
└── evaluation_results/         # Evaluation outputs
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Google Gemini API Key** - Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
3. **DriveLM Dataset** - Place in `data/` directory

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd drive-qa

# Install dependencies
pip install -r requirements.txt

# Set up environment variable
export GEMINI_API_KEY="your_api_key_here"
```

### Basic Usage

#### 1. Test Single RAG Agent Query

```bash
# Test with default parameters (Scene 1, Keyframe 1, QA Type: prediction, Serial: 1)
python scripts/run_rag_agent.py

# Test with custom parameters
python scripts/run_rag_agent.py --scene-id 2 --keyframe-id 3 --qa-type perception --qa-serial 2
```

#### 2. Run Comprehensive Evaluation

```bash
# Quick evaluation (1 scene, 1 keyframe, 2 QA pairs)
python scripts/run_eval.py --max-scenes 1 --max-keyframes 1 --max-qa-pairs 2

# Full evaluation (all scenes, 3 keyframes each, 5 QA pairs each)
python scripts/run_eval.py

# Custom scope evaluation
python scripts/run_eval.py --max-scenes 3 --max-keyframes 2 --max-qa-pairs 3
```

#### 3. Run Data Analysis

```bash
# Analyze QA patterns across all scenes
python scripts/run_analysis.py

# Generate interactive dashboard
python analysis/dashboard.py
```

## 📊 Dataset Information

The project uses the **DriveLM dataset** which contains:

- **6 Scenes** with varying driving scenarios
- **Multiple keyframes** per scene (4-8 keyframes each)
- **4 QA Types**:
  - **Perception**: Object detection and scene understanding
  - **Planning**: Route planning and decision making
  - **Prediction**: Future state prediction
  - **Behavior**: Driver behavior analysis
- **Multimodal data**:
  - Annotated images from 6 camera views
  - Vehicle movement data (speed, acceleration, position)
  - Sensor data (object detections, LiDAR/radar points)
  - Ground truth QA pairs

## 🔧 Core Components

### 1. RAG Agent (`rag/rag_agent.py`)

The main RAG agent that:
- Retrieves context from driving scenes
- Processes multimodal data (images + text)
- Generates intelligent responses using Gemini 1.5 Flash
- Handles different QA types and scenarios

### 2. Context Retriever (`rag/retrieval/context_retriever.py`)

Responsible for:
- Loading scene and keyframe data
- Extracting vehicle movement information
- Processing sensor detection data
- Generating annotated images with bounding boxes
- Providing QA pairs for evaluation

### 3. Data Loader (`parsers/data_loader.py`)

Handles:
- Loading and caching concatenated JSON data
- Scene and keyframe token management
- QA pair extraction and organization
- Data validation and error handling

### 4. Evaluation System (`scripts/run_eval.py`)

Provides:
- Comprehensive performance evaluation
- Multiple metrics (exact match, semantic similarity)
- Per-QA-type analysis
- Detailed CSV and JSON reports

## 📈 Performance Metrics

The evaluation system measures:

- **Success Rate**: Percentage of successful evaluations
- **Exact Match Rate**: Perfect answer matches
- **Semantic Similarity**: Word overlap similarity (0-1 scale)
- **Evaluation Time**: Performance timing
- **Per-QA-Type Performance**: Breakdown by question type

## 🎯 Use Cases

### Autonomous Driving Research
- Analyze driving scene understanding capabilities
- Evaluate AI reasoning in complex traffic scenarios
- Benchmark multimodal AI performance

### Education and Training
- Train autonomous driving concepts
- Demonstrate AI capabilities in driving scenarios
- Provide interactive learning experiences

### Development and Testing
- Test RAG architectures for driving applications
- Validate multimodal AI systems
- Benchmark different model approaches

## 🔍 Model Selection

The project uses **Google Gemini 1.5 Flash** for its:
- **Multimodal capabilities** (text + vision)
- **Resource efficiency** for POC development
- **Production-ready reliability**
- **Rapid development speed**

See `rag/MODEL_SELECTION.MD` for detailed analysis of alternatives including Llama 3, LLaVA, and other open-source models.

## 📝 API Reference

### RAG Agent

```python
from rag.rag_agent import RAGAgent

# Initialize agent
agent = RAGAgent(api_key="your_gemini_api_key")

# Answer a question
result = agent.answer_question(
    scene_id=1,           # Scene ID (1-6)
    keyframe_id=1,        # Keyframe ID (1-based)
    qa_type="perception", # QA type: perception, planning, prediction, behavior
    qa_serial=1           # QA pair serial number (1-based)
)

# Access results
if result["success"]:
    print(f"Question: {result['question']}")
    print(f"Model Answer: {result['model_answer']}")
    print(f"Ground Truth: {result['ground_truth_answer']}")
```

### Data Loader

```python
from parsers.data_loader import DataLoader

# Initialize data loader
dl = DataLoader()

# Load scene data
scene_data = dl.load_scene_data(scene_id=1)

# Get keyframe information
keyframe_info = dl.get_keyframe_info_for_scene(scene_id=1)

# Extract QA pairs
qa_data = dl.extract_questions_from_keyframe(scene_id=1, keyframe_id=1)
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_rag_agent.py

# Run with coverage
python -m pytest --cov=rag tests/
```

## 📊 Evaluation Results

Evaluation results are saved in `evaluation_results/` with:
- **CSV files**: Detailed results with all questions, answers, and metrics
- **JSON summaries**: Aggregated statistics and performance breakdowns
- **Log files**: Detailed execution logs for debugging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **DriveLM Dataset**: For providing comprehensive driving scene data
- **Google Gemini**: For powerful multimodal AI capabilities
- **nuScenes**: For the underlying autonomous driving dataset
- **Open Source Community**: For various tools and libraries used in this project

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation in each module
- Review the evaluation results for performance insights

---

**Note**: This project is designed for research and educational purposes. For production use in autonomous vehicles, additional safety, validation, and regulatory compliance measures would be required. 