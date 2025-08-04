# main.py
"""
Main execution script for running the FrugalGPT Difficulty Classifier on a dataset.
"""
import asyncio
import json
from tqdm.asyncio import tqdm
from typing import List, Dict, Any

from frugal_classifier import FrugalClassifier
from llm_clients import AsyncLLMClient
from config import INPUT_FILE_PATH, OUTPUT_FILE_PATH

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Loads a .jsonl file into a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Saves a list of dictionaries to a .jsonl file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"\nResults successfully saved to {file_path}")

async def main():
    """
    Main asynchronous function to run the entire classification pipeline.
    """
    llm_client = AsyncLLMClient()
    classifier = FrugalClassifier(llm_client)
    
    # Load the dataset of problems to classify
    problems_to_classify = load_jsonl(INPUT_FILE_PATH)
    results = []

    # Use tqdm.asof for an async-compatible progress bar
    for problem_data in tqdm(problems_to_classify, desc="Classifying Problems"):
        query = problem_data.get("query")
        if not query:
            continue
            
        classification_result = await classifier.classify(query)
        
        # Combine original data with the new classification results
        output_record = {**problem_data, **classification_result}
        results.append(output_record)

    # Save the final results
    save_jsonl(results, OUTPUT_FILE_PATH)
    
    # Cleanly close the client session
    await llm_client.close()

if __name__ == "__main__":
    # Execute the async main function
    asyncio.run(main())