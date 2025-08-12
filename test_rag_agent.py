#!/usr/bin/env python3
"""
Test script for the RAG Agent.
This script demonstrates how to use the RAG agent to answer questions about driving scenes.
"""

import os
from rag.rag_agent import RAGAgent
from loguru import logger

def test_rag_agent():
    """Test the RAG agent with sample data."""
    
    # Get API key from environment variable
    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GEMINI_KEY')
    if not api_key:
        logger.error("Please set GEMINI_API_KEY environment variable")
        logger.info("You can get an API key from: https://makersuite.google.com/app/apikey")
        return
    
    # Initialize the agent
    agent = RAGAgent(api_key)
    
    # Test parameters - using the first entry from the data
    scene_id = 1
    keyframe_id = 1
    qa_type = "planning"
    qa_serial = 1
    
    logger.info(f"Testing RAG Agent with:")
    logger.info(f"  Scene ID: {scene_id}")
    logger.info(f"  Keyframe ID: {keyframe_id}")
    logger.info(f"  QA Type: {qa_type}")
    logger.info(f"  QA Serial: {qa_serial}")
    
    try:
        # Get the answer
        result = agent.answer_question(scene_id, keyframe_id, qa_type, qa_serial)
        
        if result["success"]:
            logger.success("RAG Agent test successful!")
            print("\n" + "="*80)
            print("QUESTION:")
            print(result["question"])
            print("\n" + "="*80)
            print("MODEL ANSWER:")
            print(result["model_answer"])
            print("\n" + "="*80)
            print("GROUND TRUTH ANSWER:")
            print(result["ground_truth_answer"])
            print("\n" + "="*80)
            print("METADATA:")
            for key, value in result["metadata"].items():
                print(f"  {key}: {value}")
        else:
            logger.error(f"RAG Agent test failed: {result['error']}")
            
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rag_agent() 