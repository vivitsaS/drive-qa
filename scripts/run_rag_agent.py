#!/usr/bin/env python3
"""
RAG Agent Test Script

This script serves as the entry point for testing the RAG agent functionality.
It can be run with different parameters to test various scenarios.
"""

import argparse
import os
import sys
from pathlib import Path
from loguru import logger

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag.rag_agent import RAGAgent


def setup_logging():
    """Configure logging for the RAG agent test"""
    logger.add("logs/rag_test.log", rotation="10 MB", level="INFO")
    logger.info("Starting RAG Agent Test")


def validate_api_key() -> str:
    """Validate and return the API key"""
    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GEMINI_KEY')
    if not api_key:
        logger.error("Please set GEMINI_API_KEY environment variable")
        logger.info("You can get an API key from: https://makersuite.google.com/app/apikey")
        sys.exit(1)
    return api_key


def test_rag_agent(scene_id: int, keyframe_id: int, qa_type: str, qa_serial: int):
    """
    Test the RAG agent with specified parameters
    
    Args:
        scene_id: Scene ID to test
        keyframe_id: Keyframe ID to test
        qa_type: Type of question (prediction, description, etc.)
        qa_serial: QA serial number
    """
    api_key = validate_api_key()
    
    # Initialize the agent
    agent = RAGAgent(api_key)
    
    logger.info("Testing RAG Agent with:")
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


def main():
    """Main entry point for the RAG agent test script"""
    parser = argparse.ArgumentParser(description="Test the RAG Agent")
    parser.add_argument("--scene-id", type=int, default=1, help="Scene ID to test")
    parser.add_argument("--keyframe-id", type=int, default=1, help="Keyframe ID to test")
    parser.add_argument("--qa-type", type=str, default="prediction", help="QA type to test")
    parser.add_argument("--qa-serial", type=int, default=1, help="QA serial number to test")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Run the test
    test_rag_agent(args.scene_id, args.keyframe_id, args.qa_type, args.qa_serial)


if __name__ == "__main__":
    main() 