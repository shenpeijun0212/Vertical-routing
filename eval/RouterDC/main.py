# main.py
"""
Main execution script for running the RouterDC system on a dataset.
"""
import asyncio
import json
from tqdm.asyncio import tqdm
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer

from router_system import RouterDC
from llm_clients import AsyncLLMClient
import config

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Loads a .jsonl file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Saves data to a .jsonl file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"\nResults saved successfully to {file_path}")

async def main():
    """Main asynchronous function to run the RouterDC pipeline."""
    # 1. Initialize all necessary components
    feature_extractor = SentenceTransformer(config.EMBEDDING_MODEL_PATH, device=config.DEVICE)
    llm_client = AsyncLLMClient()
    
    try:
        router = RouterDC(feature_extractor, llm_client)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        await llm_client.close()
        return

    # 2. Load data
    problems_to_process = load_jsonl(config.INPUT_FILE_PATH)
    results = []

    # 3. Process each problem
    for problem_data in tqdm(problems_to_process, desc="Routing and Solving Problems"):
        query = problem_data.get("query")
        if not query:
            continue
        
        # Get the solution and metadata from the router system
        result_metadata = await router.solve_problem(query)
        
        # Combine original data with the new results
        output_record = {**problem_data, **result_metadata}
        results.append(output_record)

    # 4. Save results and clean up
    save_jsonl(results, config.OUTPUT_FILE_PATH)
    await llm_client.close()

if __name__ == "__main__":
    asyncio.run(main())