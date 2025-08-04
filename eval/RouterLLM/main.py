# main.py
"""
Main execution script for running the RouteLLM system on a dataset.
"""
import asyncio
import json
from tqdm.asyncio import tqdm
from typing import List, Dict, Any

from route_llm_system import RouteLLMSystem
from llm_clients import AsyncLLMClient
from config import INPUT_FILE_PATH, OUTPUT_FILE_PATH

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Loads a .jsonl file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: Input file not found at {file_path}")
        return []

def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Saves data to a .jsonl file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"\nResults successfully saved to {file_path}")

async def main():
    """Main asynchronous function to run the entire pipeline."""
    llm_client = AsyncLLMClient()
    system = RouteLLMSystem(llm_client)
    
    problems_to_process = load_jsonl(INPUT_FILE_PATH)
    if not problems_to_process:
        await llm_client.close()
        return
        
    results = []

    for problem_data in tqdm(problems_to_process, desc="Processing Queries with RouteLLM"):
        query = problem_data.get("query")
        if not query:
            continue
        
        # Get the final answer and metadata from the system
        result_metadata = await system.run(query)
        
        # Combine original data with the new results
        output_record = {**problem_data, **result_metadata}
        results.append(output_record)

    save_jsonl(results, OUTPUT_FILE_PATH)
    await llm_client.close()

if __name__ == "__main__":
    asyncio.run(main())