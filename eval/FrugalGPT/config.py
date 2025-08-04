# config.py
"""
Configuration file for the FrugalGPT-based Difficulty Classifier.

This file contains all settings, including model identifiers, API configurations,
and the specific prompts used to instruct the LLMs.
"""

import os

# --- Security Note ---
# NEVER hardcode API keys directly in the code.
# Use environment variables for security.
# Example: os.getenv("OPENAI_API_KEY")
# In your terminal, you would set this with:
# export OPENAI_API_KEY='your_key_here'

API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    # Add other providers as needed
    "default": os.getenv("DEFAULT_API_KEY")
}

# --- Model Configuration ---

# Define the "expert" models. These are typically faster, cheaper models.
# The structure is: "custom_name": {"provider": "provider_key", "model_id": "api_model_name"}
EXPERT_MODELS = {
    "expert_gpt3_5": {"provider": "openai", "model_id": ""},
    "expert_claude_haiku": {"provider": "anthropic", "model_id": ""},
    # You can add more models, even from the same provider
    # "expert_gpt4_mini": {"provider": "openai", "model_id": "gpt-4-turbo"},
}

# Define the "judge" model. This is typically a more powerful, capable model.
JUDGE_MODEL = {}


# --- Prompt Engineering ---

# The prompt template for the expert models.
# It should instruct the model to provide a direct, one-word classification.
EXPERT_PROMPT_TEMPLATE = """
Classify the following math problem's difficulty. The user is a middle school student.
Your answer must be a single word: either 'easy' or 'difficult'.

Problem:
"{query}"

Classification:
"""

# The prompt template for the judge model.
# It receives the original query and the various opinions from the experts.
# Its task is to make a final, reasoned decision.
JUDGE_PROMPT_TEMPLATE = """
You are a master evaluator of AI model outputs.
Multiple AI "expert" models have provided their opinion on the difficulty of a math problem.
Your task is to analyze their opinions and the problem itself to make a final, definitive classification.
Your final answer must be a single word: either 'easy' or 'difficult'.

---
Original Problem:
"{query}"
---
Expert Opinions:
{expert_opinions}
---

Based on your expert analysis, the final difficulty classification is:
"""

# Input file containing the problems to be classified
INPUT_FILE_PATH = ""

# Output file where the final classifications will be saved
OUTPUT_FILE_PATH = ""