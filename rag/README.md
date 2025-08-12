# RAG Agent Setup

This directory contains a minimal RAG (Retrieval-Augmented Generation) agent setup for autonomous driving question answering.

## Overview

The RAG agent uses:
- **Gemini Flash 1.5** as the multimodal model
- **ContextRetriever** functions as tools to fetch relevant data
- Scene, keyframe, and QA pair information to generate answers

## Files

- `rag_agent.py` - Main RAG agent class
- `test_rag_agent.py` - Test script to demonstrate usage
- `retrieval/context_retriever.py` - Existing retriever functions (already implemented)

## Usage

### 1. Setup API Key

Set your Gemini API key as an environment variable:
```bash
export GEMINI_API_KEY="your_api_key_here"
```

Get an API key from: https://makersuite.google.com/app/apikey

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Basic Usage

```python
from rag.rag_agent import RAGAgent

# Initialize agent
agent = RAGAgent(api_key="your_api_key")

# Answer a question
result = agent.answer_question(
    scene_id="cc8c0bf57f984915a77078b10eb33198",
    keyframe_id=1,
    qa_type="perception",
    qa_serial=1
)

if result["success"]:
    print(f"Question: {result['question']}")
    print(f"Answer: {result['model_answer']}")
    print(f"Ground Truth: {result['ground_truth_answer']}")
```

### 4. Run Test

```bash
python rag/test_rag_agent.py
```

## Input Parameters

- `scene_id`: Scene identifier (string)
- `keyframe_id`: Keyframe identifier (1-based integer)
- `qa_type`: Type of QA (e.g., "perception", "prediction")
- `qa_serial`: QA pair serial number (1-based integer)

## Available Tools

The agent has access to these tools from `context_retriever.py`:

1. **Context Data**: Scene and keyframe information
2. **Vehicle Data**: Movement data (speed, acceleration, position)
3. **Sensor Data**: Object detections, LiDAR/radar points
4. **Annotated Images**: Visual data with bounding boxes

## Output

The agent returns a dictionary with:
- `success`: Boolean indicating success/failure
- `question`: The original question from annotations
- `model_answer`: Generated answer from Gemini
- `ground_truth_answer`: Answer from annotations (for evaluation)
- `metadata`: Information about available data sources

## Next Steps

For evaluation, you can compare `model_answer` with `ground_truth_answer` using metrics like:
- BLEU score
- ROUGE score
- Semantic similarity
- Human evaluation 