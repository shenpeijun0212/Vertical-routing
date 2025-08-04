# route_llm_system.py
"""
Implements the core logic for the RouteLLM system.
It uses one LLM to route a query to another specialized expert LLM.
"""
import asyncio
from typing import Dict, Any

from llm_clients import AsyncLLMClient
from config import (
    ROUTER_LLM,
    EXPERT_MODELS,
    DEFAULT_EXPERT_KEY,
    ROUTER_PROMPT_TEMPLATE,
    EXPERT_PROMPT_TEMPLATE,
)

class RouteLLMSystem:
    """
    Orchestrates the router-expert LLM workflow.
    """
    def __init__(self, llm_client: AsyncLLMClient):
        self.client = llm_client
        self.expert_keys = list(EXPERT_MODELS.keys())

    def _parse_routing_decision(self, response: str) -> str:
        """
        Parses the router's text response to find a valid expert key.

        Args:
            response (str): The raw text output from the router LLM.

        Returns:
            str: A valid expert key, or the default key if parsing fails.
        """
        cleaned_response = response.lower().strip().replace("'", "").replace('"', "")
        for key in self.expert_keys:
            if key in cleaned_response:
                return key
        # If no valid key is found in the response, use the default
        return DEFAULT_EXPERT_KEY

    async def run(self, query: str) -> Dict[str, Any]:
        """
        Executes the full RouteLLM pipeline for a single query.
        1. Queries the router LLM.
        2. Parses the decision.
        3. Queries the chosen expert LLM for the final answer.

        Args:
            query (str): The user's input problem or question.

        Returns:
            Dict[str, Any]: A dictionary containing the final answer and routing metadata.
        """
        # 1. Get routing decision from the Router LLM
        router_prompt = ROUTER_PROMPT_TEMPLATE.format(query=query)
        raw_routing_decision = await self.client.query(
            provider=ROUTER_LLM["provider"],
            model_id=ROUTER_LLM["model_id"],
            prompt=router_prompt
        )

        # 2. Parse the decision to get a valid expert key
        chosen_expert_key = self._parse_routing_decision(raw_routing_decision)
        chosen_expert = EXPERT_MODELS[chosen_expert_key]
        
        print(f"Query: '{query[:40]}...' -> Route: '{chosen_expert_key}' -> Expert: {chosen_expert['name']}")

        # 3. Query the chosen Expert LLM for the final answer
        expert_prompt = EXPERT_PROMPT_TEMPLATE.format(query=query)
        final_answer = await self.client.query(
            provider=chosen_expert["provider"],
            model_id=chosen_expert["model_id"],
            prompt=expert_prompt
        )
        
        return {
            "routing_decision": chosen_expert_key,
            "expert_used": chosen_expert['name'],
            "final_answer": final_answer,
            "raw_router_output": raw_routing_decision
        }