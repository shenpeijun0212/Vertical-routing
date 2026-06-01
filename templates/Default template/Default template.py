# -*- coding: utf-8 -*-
"""
Standalone Example of the Default Template.

This script demonstrates the logic of the "Default Template" as described
in the "Vertical Routing" paper. This template is a general-purpose strategy
that doesn't rely on task-specific structures. It is a self-contained
example that uses mock objects for models and tokenizers to focus purely
on the template's workflow.

Workflow:
1.  **Initial Generation (Large Model)**: A mock "large model" generates the
    first N tokens of the response, which are considered the most critical.
2.  **Continuation (Small Model)**: A mock "small model" takes the initial
    prompt plus the large model's output and continues generating the
    response until it's complete.
"""
import os
import re
import time
from typing import Dict, Any, Tuple, List

# --- Mock Objects for Demonstration ---
# In a real scenario, these would be imported from the transformers library.
class MockPreTrainedModel:
    """A mock class to simulate a Hugging Face model."""
    def __init__(self, model_type="large"):
        self.model_type = model_type

    def generate(self, input_ids, **kwargs) -> List[List[int]]:
        # Simulate model generation based on the length of the input prompt.
        # The small model's prompt will be longer as it includes the large model's output.
        if len(input_ids[0]) < 100: # Arbitrary length to distinguish the initial call
            # This is the initial generation prompt for the large model
            return [[0] * 20] # Return a fixed-length list of mock token IDs for the first part
        else:
            # This is the continuation prompt for the small model
            return [[1] * 50] # Return a different list of mock token IDs for the second part
            
    def to(self, device):
        # Mock device placement
        return self

class MockTokenizer:
    """A mock class to simulate a Hugging Face tokenizer."""
    def __call__(self, text_list, return_tensors="pt"):
        # Simulate tokenization by returning the length of the text as mock IDs
        return {"input_ids": [[i for i in range(len(text_list[0]))]]}

    def encode(self, text, add_special_tokens=False):
        # Simulate encoding
        return [123]

    def decode(self, token_ids, skip_special_tokens=True):
        # Simulate decoding based on the mock token IDs from the model
        if token_ids and token_ids[0] == 0:
            return "Plan:\nStep 1: Calculate the total items. "
        elif token_ids and token_ids[0] == 1:
            return "Step 2: Subtract the used items.\nSolution:\nTotal items = 10 * 5 = 50. Used items = 50 - 5 = 45. #### 45 <end>"
        return "".join([str(t) for t in token_ids])

# Instantiate the mock tokenizer globally
MOCK_TOKENIZER = MockTokenizer()

# --- 1. 配置模块 (Configuration Module) ---
class Config:
    """Minimal configuration for the template."""
    DEVICE: str = "cpu"
    # The number of initial tokens for the large model to generate
    LARGE_MODEL_TOKENS: int = 40
    # The maximum number of tokens for the small model to continue with
    SMALL_MODEL_TOKENS: int = 150
    TEMPERATURE: float = 0.1
    TOP_P: float = 0.9
    DO_SAMPLE: bool = False


# --- 2. 模型管理模块 (Model Management Module) ---
class MockModelManager:
    """A mock model manager that provides mock models and tokenizers."""
    def __init__(self, config: Config):
        self.config = config
        print("Loading MOCK models and tokenizers...")
        self.large_model = MockPreTrainedModel(model_type="large")
        self.small_model = MockPreTrainedModel(model_type="small")
        self.large_tokenizer = MOCK_TOKENIZER
        self.small_tokenizer = MOCK_TOKENIZER
        print("✅ Mock models and tokenizers are ready.")


# --- 3. 模板定义模块 (Template Definition Module) ---
class BaseTemplate:
    """Base class for all routing templates."""
    def __init__(self, model_manager: MockModelManager, config: Config):
        self.mm = model_manager
        self.config = config
        self.name = self.__class__.__name__

    def _generate(self, model: MockPreTrainedModel, tokenizer: MockTokenizer, prompt: str, max_new_tokens: int) -> Tuple[str, int]:
        """Generic text generation function using mock objects."""
        inputs = tokenizer([prompt], return_tensors="pt")
        
        stop_sequence = "<end>"
        
        generated_ids = model.generate(inputs['input_ids'])
        response_full = tokenizer.decode(generated_ids[0])
        response = response_full.split(stop_sequence)[0].strip()

        num_tokens = len(response.split())
        return response, num_tokens

    def run(self, query: str) -> Tuple[str, int, int]:
        """This method should be implemented by subclasses."""
        raise NotImplementedError

class DefaultTemplate(BaseTemplate):
    """
    Implements the Default Template: Large model generates the first part,
    small model generates the rest.
    """
    def run(self, query: str) -> Tuple[str, int, int]:
        """Executes the sequential generation workflow."""
        print("--- Executing Default Template ---")
        
        # The full prompt including few-shot examples (if any) and the new query
        full_prompt = f"Question: {query}\nAnswer:"

        # --- Stage 1: Large Model generates the beginning ---
        print("\n[Stage 1] Generating initial response with Large Model...")
        large_model_output, large_model_tokens = self._generate(
            model=self.mm.large_model,
            tokenizer=self.mm.large_tokenizer,
            prompt=full_prompt,
            max_new_tokens=self.config.LARGE_MODEL_TOKENS
        )
        print(f"Initial part from Large Model: '{large_model_output}'")

        # --- Stage 2: Small Model continues the generation ---
        print("\n[Stage 2] Continuing generation with Small Model...")
        # The prompt for the small model includes the large model's output
        continuation_prompt = full_prompt + large_model_output
        
        small_model_output, small_model_tokens = self._generate(
            model=self.mm.small_model,
            tokenizer=self.mm.small_tokenizer,
            prompt=continuation_prompt,
            max_new_tokens=self.config.SMALL_MODEL_TOKENS
        )
        print(f"Continuation from Small Model: '{small_model_output}'")

        total_response = large_model_output + small_model_output
        return total_response, large_model_tokens, small_model_tokens


# --- 4. 主执行函数 (Main Execution Function) ---
def main():
    """Main function to demonstrate the DefaultTemplate."""
    print("🚀 Initializing standalone Default Template demo...")
    
    # Configuration and mock manager setup
    config = Config()
    mock_manager = MockModelManager(config)
    
    # Instantiate the template
    default_template = DefaultTemplate(mock_manager, config)
    
    # Define a sample query
    sample_query = "A factory produces 10 boxes of pencils per day, with 5 pencils in each box. If a student takes 5 pencils, how many are left?"
    print(f"\nSample Query: \"{sample_query}\"")
    
    # Run the template
    final_response, initial_tokens, continuation_tokens = default_template.run(sample_query)
    
    # Print the final results
    print("\n--- ✅ Final Combined Output ---")
    print(final_response)
    print("\n--- 📊 Summary ---")
    print(f"Tokens from Large Model (Initial Part): {initial_tokens}")
    print(f"Tokens from Small Model (Continuation): {continuation_tokens}")
    print(f"Total Generated Tokens: {initial_tokens + continuation_tokens}")

if __name__ == "__main__":
    main()
