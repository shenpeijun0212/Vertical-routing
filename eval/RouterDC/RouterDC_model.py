# router_system.py
"""
Implements the core logic for the RouterDC system.
It uses a local MLP model to route a query to either a simple or complex expert LLM.
"""
import torch
from sentence_transformers import SentenceTransformer
import os

from mlp_router_model import MLP
from llm_clients import AsyncLLMClient
import config

class RouterDC:
    """
    A system that routes problems to different expert LLMs based on
    a difficulty classification from a local router model.
    """
    def __init__(self, feature_extractor: SentenceTransformer, llm_client: AsyncLLMClient):
        self.device = config.DEVICE
        self.feature_extractor = feature_extractor
        self.llm_client = llm_client
        self.router_model = self._load_router_model()
        self.label_map_rev = {0: 'difficult', 1: 'easy'}

    def _load_router_model(self) -> MLP:
        """Loads the pre-trained MLP router model from disk."""
        if not os.path.exists(config.ROUTER_MODEL_PATH):
            raise FileNotFoundError(
                f"Router model not found at {config.ROUTER_MODEL_PATH}. "
                "Please train the MLP model first and place it at the correct path."
            )
            
        print(f"Loading router model from {config.ROUTER_MODEL_PATH}...")
        model = MLP(
            input_dim=config.ROUTER_INPUT_DIM,
            hidden_dim=config.ROUTER_HIDDEN_DIM,
            output_dim=config.ROUTER_OUTPUT_DIM,
        ).to(self.device)
        model.load_state_dict(torch.load(config.ROUTER_MODEL_PATH, map_location=self.device))
        model.eval()
        print("Router model loaded successfully.")
        return model

    def _get_routing_decision(self, query: str) -> str:
        """
        Uses the local MLP router to classify the query's difficulty.
        
        Args:
            query (str): The input problem text.

        Returns:
            str: The predicted difficulty ('easy' or 'difficult').
        """
        embedding = self.feature_extractor.encode(
            query,
            convert_to_tensor=True
        ).to(self.device)
        
        with torch.no_grad():
            output = self.router_model(embedding.unsqueeze(0))
            prediction_idx = torch.argmax(output, dim=1).item()
            
        return self.label_map_rev[prediction_idx]

    async def solve_problem(self, query: str) -> dict:
        """
        Executes the full RouterDC pipeline for a single problem.
        1. Gets a routing decision.
        2. Routes to the appropriate expert LLM.
        3. Returns the solution and metadata.

        Args:
            query (str): The input problem text.

        Returns:
            dict: A dictionary with the result and routing metadata.
        """
        # 1. Classify difficulty to decide the route
        routing_decision = self._get_routing_decision(query)
        
        # 2. Select the expert model based on the decision
        if routing_decision == 'easy':
            expert = config.SIMPLE_EXPERT_MODEL
        else:
            expert = config.COMPLEX_EXPERT_MODEL
            
        print(f"Query routed to '{routing_decision}' -> Using expert: {expert['name']}")
        
        # 3. Query the selected expert to get the solution
        prompt = config.EXPERT_SOLVER_PROMPT_TEMPLATE.format(query=query)
        solution = await self.llm_client.query(
            provider=expert["provider"],
            model_id=expert["model_id"],
            prompt=prompt
        )
        
        return {
            "routed_to": routing_decision,
            "expert_used": expert['name'],
            "expert_solution": solution
        }