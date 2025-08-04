# frugal_classifier.py
"""
Implements the core logic for the FrugalGPT classification strategy.
"""
import asyncio
from typing import Dict, Any

from llm_clients import AsyncLLMClient
from config import (
    EXPERT_MODELS,
    JUDGE_MODEL,
    EXPERT_PROMPT_TEMPLATE,
    JUDGE_PROMPT_TEMPLATE,
)

class FrugalClassifier:
    """
    Orchestrates a multi-LLM classification strategy involving "experts" and a "judge".
    """
    def __init__(self, llm_client: AsyncLLMClient):
        self.client = llm_client

    async def _get_expert_opinions(self, query: str) -> Dict[str, str]:
        """
        Queries all expert models concurrently to get their classifications.

        Args:
            query (str): The input problem text.

        Returns:
            Dict[str, str]: A dictionary mapping expert names to their responses.
        """
        tasks = []
        for name, model_info in EXPERT_MODELS.items():
            prompt = EXPERT_PROMPT_TEMPLATE.format(query=query)
            task = self.client.query(
                provider=model_info["provider"],
                model_id=model_info["model_id"],
                prompt=prompt
            )
            tasks.append(task)
        
        # Execute all API calls in parallel for efficiency
        expert_responses = await asyncio.gather(*tasks)
        
        return {name: response for name, response in zip(EXPERT_MODELS.keys(), expert_responses)}

    async def _get_judge_decision(self, query: str, opinions: Dict[str, str]) -> str:
        """
        Queries the judge model with the problem and expert opinions to get a final decision.

        Args:
            query (str): The input problem text.
            opinions (Dict[str, str]): The collected opinions from the experts.

        Returns:
            str: The final classification from the judge model.
        """
        # Format the expert opinions for inclusion in the judge's prompt
        opinions_str = "\n".join([f"- {name}: {opinion}" for name, opinion in opinions.items()])
        
        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(query=query, expert_opinions=opinions_str)
        
        judge_info = list(JUDGE_MODEL.values())[0]
        
        decision = await self.client.query(
            provider=judge_info["provider"],
            model_id=judge_info["model_id"],
            prompt=judge_prompt
        )
        return decision

    async def classify(self, problem_query: str) -> Dict[str, Any]:
        """
        Performs the full FrugalGPT classification for a single problem.

        Args:
            problem_query (str): The text of the math problem.

        Returns:
            Dict[str, Any]: A dictionary containing the final decision and the expert opinions.
        """
        print(f"Processing query: '{problem_query[:50]}...'")
        
        # 1. Get opinions from all experts
        expert_opinions = await self._get_expert_opinions(problem_query)
        print(f"  - Expert Opinions: {expert_opinions}")
        
        # 2. Get the final decision from the judge
        final_decision = await self._get_judge_decision(problem_query, expert_opinions)
        print(f"  - Judge's Decision: {final_decision}")
        
        # Clean up the final answer to be strictly 'easy' or 'difficult'
        cleaned_decision = 'easy' if 'easy' in final_decision.lower() else 'difficult'
        
        return {
            "final_difficulty": cleaned_decision,
            "expert_opinions": expert_opinions
        }