# llm_clients.py
"""
Asynchronous client to handle API calls to various LLM providers.
"""
import httpx
from typing import Dict, Any

from config import API_KEYS, API_ENDPOINTS

class AsyncLLMClient:
    """An asynchronous client for querying various LLM APIs."""
    def __init__(self, timeout: int = 120):
        self.client = httpx.AsyncClient(timeout=timeout)

    async def query(self, provider: str, model_id: str, prompt: str) -> str:
        """
        Sends a prompt to a specified LLM and returns the response text.

        Args:
            provider (str): The API provider (e.g., 'openai', 'anthropic').
            model_id (str): The specific model ID for the API.
            prompt (str): The user-role prompt to send to the model.

        Returns:
            str: The text content of the model's response.
        """
        endpoint = API_ENDPOINTS.get(provider)
        api_key = API_KEYS.get(provider)

        if not endpoint or not api_key:
            return "Error: Provider or API key not configured."

        headers = {"Content-Type": "application/json"}
        payload: Dict[str, Any] = {}

        if provider == "openai":
            headers["Authorization"] = f"Bearer {api_key}"
            payload = {"model": model_id, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1500}
        elif provider == "anthropic":
            headers["x-api-key"] = api_key
            headers["anthropic-version"] = "2023-06-01"
            payload = {"model": model_id, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1500}

        try:
            response = await self.client.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            response_data = response.json()
            
            if provider == "openai":
                return response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            elif provider == "anthropic":
                return response_data.get("content", [{}])[0].get("text", "").strip()
            
            return "Error: Response parsing not implemented for this provider."
        except httpx.HTTPStatusError as e:
            return f"Error: HTTP {e.response.status_code} - {e.response.text}"
        except Exception as e:
            return f"Error: An unexpected error occurred - {str(e)}"

    async def close(self):
        """Closes the HTTP client session."""
        await self.client.aclose()