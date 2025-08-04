# llm_clients.py
"""
Asynchronous client to handle API calls to various LLM providers.
This abstracts away the specific request/response formats of each API.
"""
import httpx
from typing import Dict, Any

from config import API_KEYS, API_ENDPOINTS

class AsyncLLMClient:
    """
    An asynchronous client for querying various LLM APIs.
    """
    def __init__(self, timeout: int = 60):
        self.client = httpx.AsyncClient(timeout=timeout)

    async def query(self, provider: str, model_id: str, prompt: str) -> str:
        """
        Sends a prompt to a specified LLM and returns the response text.

        Args:
            provider (str): The API provider (e.g., 'openai').
            model_id (str): The specific model ID for the API.
            prompt (str): The formatted prompt to send to the model.

        Returns:
            str: The text content of the model's response.
        """
        endpoint = API_ENDPOINTS.get(provider)
        api_key = API_KEYS.get(provider)

        if not endpoint or not api_key:
            return "Error: Provider or API key not configured."

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        # --- Provider-Specific Payload Formatting ---
        # This section must be adapted for the specific APIs you use.
        payload: Dict[str, Any] = {}
        if provider == "openai":
            # This payload is for older completion models like gpt-3.5-turbo-instruct
            payload = {
                "model": model_id,
                "prompt": prompt,
                "max_tokens": 10,
                "temperature": 0.0,
            }
        elif provider == "anthropic":
            headers["x-api-key"] = api_key # Anthropic uses a different header
            headers.pop("Authorization", None)
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10,
            }
        # Add other providers (elif provider == 'google': ...) here.
        
        try:
            response = await self.client.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            response_data = response.json()
            
            # --- Provider-Specific Response Parsing ---
            if provider == "openai":
                return response_data.get("choices", [{}])[0].get("text", "").strip()
            elif provider == "anthropic":
                return response_data.get("content", [{}])[0].get("text", "").strip()
            
            return "Error: Response parsing not implemented for this provider."

        except httpx.HTTPStatusError as e:
            return f"Error: HTTP Error {e.response.status_code} - {e.response.text}"
        except Exception as e:
            return f"Error: An unexpected error occurred - {str(e)}"

    async def close(self):
        """Closes the HTTP client session."""
        await self.client.aclose()