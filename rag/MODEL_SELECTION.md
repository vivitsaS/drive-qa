# Model Selection for DriveLM RAG Pipeline

## Overview
This document justifies the model selection for our multi-modal RAG system that processes autonomous driving data including text, structured data, and camera images.

## Selected Models

### Vision-Language Model: LLaVA-1.5-7B
**Primary model for multi-modal reasoning**

**Architecture:**
- Base: Vicuna-7B (fine-tuned Llama 2)
- Vision encoder: CLIP ViT-L/14
- Vision-language connector: Simple linear projection layer
- Total parameters: ~7B

**Justification:**
- **Multi-modal capability**: Native support for both text and image inputs, essential for processing camera data + metadata
- **Performance**: Strong performance on VQA benchmarks, particularly for spatial reasoning tasks relevant to driving scenarios
- **Size efficiency**: 7B parameters provide good balance between capability and inference speed
- **Open source**: MIT license allows modification and deployment
- **Driving relevance**: Performs well on spatial understanding tasks like "Is the car in front braking?" which require visual + contextual reasoning
- **Hardware requirements**: Runs efficiently on single GPU (16GB+ VRAM)

### Text Model: Meta-Llama-3-8B-Instruct
**Fallback for text-only queries and structured reasoning**

**Architecture:**
- Transformer decoder with standard multi-head attention
- RMSNorm normalization and SwiGLU activation function
- Grouped Query Attention (GQA) for efficiency
- Total parameters: 8B

**Justification:**
- **Superior reasoning**: Llama 3 shows significantly better performance on reasoning benchmarks
- **Instruction following**: Excellent at following complex multi-step instructions for data analysis
- **Safety and alignment**: Better aligned for safe, helpful responses in autonomous driving context
- **Multilingual capability**: Strong performance across languages (useful for international datasets)
- **Context length**: 8K context window for processing longer driving sequences
- **Open source**: Custom license allowing commercial use

## Architecture Decision Rationale

### Why LLaVA over alternatives:
- **vs CogVLM**: LLaVA has better community support and clearer documentation for fine-tuning
- **vs GPT-4V**: Open source requirement, cost considerations for batch processing
- **vs BLIP-2**: LLaVA shows superior performance on complex reasoning tasks
- **vs Flamingo**: Better availability and easier deployment

### Why Llama 3 over alternatives:
- **vs Mistral-7B**: Llama 3-8B shows superior performance on reasoning tasks and instruction following
- **vs Llama 2-7B**: Significant improvements in reasoning, code understanding, and multilingual capabilities
- **vs CodeLlama**: Llama 3 maintains code capabilities while being better at general reasoning
- **vs Falcon**: Better performance on complex reasoning benchmarks relevant to driving scenarios

## Pipeline Strategy

### Multi-modal queries (text + image):
LLaVA-1.5-7B processes both camera images and structured context together

### Text-only queries:
Llama-3-8B-Instruct handles structured data analysis and metadata reasoning

### Hybrid approach:
- Use LLaVA for visual reasoning: "What objects are visible in the camera?"
- Use Llama 3 for analytical tasks: "Analyze velocity patterns in the last 5 seconds"
- Combine outputs for complex scenarios: "Is the car ahead slowing down based on visual cues and velocity data?"

## Performance Considerations

### Memory Requirements:
- LLaVA-1.5-7B: ~16GB VRAM for inference
- Llama-3-8B: ~10GB VRAM for inference
- Total system: 26GB VRAM recommended (can run sequentially on 16GB)

### Inference Speed:
- LLaVA: ~2-3 tokens/second on RTX 4090
- Llama 3: ~12-18 tokens/second on RTX 4090
- Acceptable for batch processing and research use

### Accuracy Trade-offs:
- Prioritized model capability over pure speed
- 7B models provide good balance for research/development phase
- Can upgrade to larger models (13B/34B) if computational resources allow

## Implementation Notes

### Model Loading:
- Use Hugging Face Transformers for consistent interface
- Implement model caching to avoid reloading
- Support for quantization (4-bit/8-bit) to reduce memory usage

### Context Management:
- LLaVA: Max context ~2048 tokens + image
- Llama 3: Max context ~8192 tokens
- Implement context truncation strategy for long driving sequences

## Future Considerations

### Potential Upgrades:
- **LLaVA-1.6**: When available, offers improved performance
- **Llama-3-70B**: For significantly better reasoning if compute allows
- **Custom fine-tuning**: On DriveLM data for domain-specific performance

### Evaluation Metrics:
- Visual reasoning accuracy on driving scenarios
- Structured data analysis correctness
- Response coherence and factual accuracy
- Inference latency for real-time applications