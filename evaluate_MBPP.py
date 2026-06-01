# -*- coding: utf-8 -*-
"""
Standalone Example for MBPP Dataset Evaluation using the Default Template.

This script demonstrates how the "Default Template" from the "Vertical
Routing" paper can be applied to code generation tasks like MBPP.
It is a self-contained example using mock objects to simulate the workflow.

Workflow for Code Generation:
1.  **Initial Code Generation (Large Model)**: A mock "large model" generates
    the beginning of the Python function, including the signature and docstring,
    which is considered the most critical part for setting the context.
2.  **Code Completion (Small Model)**: A mock "small model" takes the initial
    code snippet and completes the function's logic.
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
        if len(input_ids[0]) < 150: # Arbitrary length to distinguish the initial call
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
            return 'def find_max(arr):\n    """\n    Finds the maximum element in a list of numbers.\n    """\n    if not arr:\n        return None'
        elif token_ids and token_ids[0] == 1:
            return '\n    max_val = arr[0]\n    for x in arr[1:]:\n        if x > max_val:\n            max_val = x\n    return max_val'
        return "".join([str(t) for t in token_ids])

# Instantiate the mock tokenizer globally
MOCK_TOKENIZER = MockTokenizer()

# --- 1. 配置模块 (Configuration Module) ---
class Config:
    """Minimal configuration for the template."""
    DEVICE: str = "cpu"
    # The number of initial tokens for the large model to generate
    LARGE_MODEL_TOKENS: int = 60
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
        
        # In code generation, we might not have a simple stop sequence like "<end>"
        # The model should learn to stop after completing the function.
        # For this mock, the tokenizer's decode handles the simulated output.
        
        generated_ids = model.generate(inputs['input_ids'])
        response = tokenizer.decode(generated_ids[0])

        num_tokens = len(response.split()) # A simple token count approximation
        return response, num_tokens

    def run(self, query: str) -> Tuple[str, int, int]:
        """This method should be implemented by subclasses."""
        raise NotImplementedError

class DefaultTemplate(BaseTemplate):
    """
    Implements the Default Template for code generation.
    """
    def run(self, query: str) -> Tuple[str, int, int]:
        """Executes the sequential code generation workflow."""
        print("--- Executing Default Template for Code Generation ---")
        
        # A prompt suitable for code generation tasks
        full_prompt = (
            "You are an expert Python programmer. Please complete the following function based on the provided description.\n\n"
            f'"""\n{query}\n"""\n'
        )

        # --- Stage 1: Large Model generates the function signature and start ---
        print("\n[Stage 1] Generating initial code with Large Model...")
        large_model_output, large_model_tokens = self._generate(
            model=self.mm.large_model,
            tokenizer=self.mm.large_tokenizer,
            prompt=full_prompt,
            max_new_tokens=self.config.LARGE_MODEL_TOKENS
        )
        print(f"Initial code from Large Model:\n---\n{large_model_output}\n---")

        # --- Stage 2: Small Model continues the generation ---
        print("\n[Stage 2] Completing the code with Small Model...")
        continuation_prompt = full_prompt + large_model_output
        
        small_model_output, small_model_tokens = self._generate(
            model=self.mm.small_model,
            tokenizer=self.mm.small_tokenizer,
            prompt=continuation_prompt,
            max_new_tokens=self.config.SMALL_MODEL_TOKENS
        )
        print(f"Completed code from Small Model:\n---\n{small_model_output}\n---")

        total_response = large_model_output + small_model_output
        return total_response, large_model_tokens, small_model_tokens


# --- 4. 主执行函数 (Main Execution Function) ---
def main():
    """Main function to demonstrate the DefaultTemplate for MBPP."""
    print("🚀 Initializing standalone MBPP evaluation demo...")
    
    # Configuration and mock manager setup
    config = Config()
    mock_manager = MockModelManager(config)
    
    # Instantiate the template
    default_template = DefaultTemplate(mock_manager, config)
    
    # Define a sample MBPP-style query
    sample_query = "Write a function to find the maximum element in a list of numbers."
    print(f"\nSample Task Description: \"{sample_query}\"")
    
    # Run the template
    final_code, initial_tokens, continuation_tokens = default_template.run(sample_query)
    
    # Print the final results
    print("\n--- ✅ Final Generated Code ---")
    print(final_code)
    print("\n--- 📊 Summary ---")
    print(f"Tokens from Large Model (Initial Code): {initial_tokens}")
    print(f"Tokens from Small Model (Code Completion): {continuation_tokens}")
    print(f"Total Generated Tokens: {initial_tokens + continuation_tokens}")
    print("\nNote: In a real evaluation, this generated code would be executed against test cases to calculate metrics like pass@k.")

if __name__ == "__main__":
    main()
