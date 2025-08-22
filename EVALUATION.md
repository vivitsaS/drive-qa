# RAG Agent Evaluation Guide

## Problem with current metrics


### **1. Exact Match (0% success rate)**
- **Problem**: Compares full strings, penalizing detailed explanations
- **Example**: 
  - Ground Truth: "No."
  - Model Answer: "No. The orange roadblock is located on the shoulder/side of the road, outside the ego vehicle's lane. It does not pose a threat to the ego vehicle's path."
- **Reality**: Model provides comprehensive reasoning, which is better than simple answers

### **2. Semantic Similarity (0.05 average)**
- **Problem**: Uses simple word overlap (Jaccard similarity), which penalizes detailed explanations
- **Example**: Model's detailed answer contains many more relevant words than ground truth
- **Reality**: More comprehensive answers should score higher, not lower

### **3. Ground Truth Quality Issues**
- Some ground truth answers appear incorrect or overly generic
- Model's answers are more precise and contextually appropriate
