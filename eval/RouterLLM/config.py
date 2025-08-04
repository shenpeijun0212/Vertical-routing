# config.py
"""
Configuration file for the RouteLLM system.

This file centralizes all settings, including the router LLM, the specialized
expert LLMs, their prompts, and API configurations.
"""

import os

# --- Security Note ---
# Load API keys from environment variables for security.
# In your terminal: export OPENAI_API_KEY='your_key_here'
API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
}

# --- Model Configuration ---

# 1. THE ROUTER LLM
# This model's only job is to classify the query and choose an expert.
# It should be fast and cheap (e.g., Claude 3 Haiku, Gemini 1.5 Flash, GPT-3.5-Turbo).
ROUTER_LLM = {
    "provider": "anthropic",
    "model_id": "claude-3-haiku-20240307"
}

# 2. THE EXPERT LLMs
# Define the specialized experts the router can choose from. The keys ('math_solver',
# 'general_qa') are the routing targets. The router must output one of these keys.
EXPERT_MODELS = {
    "math_solver": {
        "name": "GPT-4o (Math Expert)",
        "provider": "openai",
        "model_id": "gpt-4o"
    },
    "general_qa": {
        "name": "Claude 3 Sonnet (General QA)",
        "provider": "anthropic",
        "model_id": "claude-3-sonnet-20240229"
    }
}

# 3. FALLBACK MECHANISM
# If the router's output is unclear, a default expert will be used.
# This key MUST exist in the EXPERT_MODELS dictionary.
DEFAULT_EXPERT_KEY = "math_solver"


# --- Prompt Engineering ---

# 1. ROUTER PROMPT
# This prompt is critical. It constrains the router to only output the name
# of the chosen expert.
ROUTER_PROMPT_TEMPLATE = """
You are a highly intelligent task routing system. Your job is to analyze the user's query and decide which expert is best suited to answer it.

Here are the available experts:
- 'math_solver': Best for mathematical problems, word problems, and logic puzzles that require step-by-step reasoning and calculations.
- 'general_qa': Best for general knowledge questions, summarization, or questions that do not involve complex math.

Based on the query below, which expert should be used?
Respond with ONLY the expert's name ('math_solver' or 'general_qa'). Do not add any other text or punctuation.

Query: "{query}"

Expert:
"""

# 2. EXPERT PROMPT
# This is the prompt that will be sent to the CHOSEN expert to generate the final answer.
EXPERT_PROMPT_TEMPLATE = """
You are a world-class expert specializing in your field.
Please provide a clear, accurate, and step-by-step answer to the following query.
For math problems, provide the final numerical answer inside #### markers at the end. For example: ####123####.

Query: "{query}"

Answer:
"""

# --- API and File Path Configuration ---
API_ENDPOINTS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "anthropic": "https://api.anthropic.com/v1/messages",
}
INPUT_FILE_PATH = "/gemini/code/RouteAgent/KNN/GSM8K_random_500.jsonl"
OUTPUT_FILE_PATH = "routellm_results.jsonl"