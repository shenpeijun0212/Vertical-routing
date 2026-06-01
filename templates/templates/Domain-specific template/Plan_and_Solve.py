# -*- coding: utf-8 -*-
"""
Standalone Example of the Plan-and-Solve (PS) Domain-Specific Template.

This script demonstrates the logic of the "Plan-and-Solve" (PS) template
as described in the "Vertical Routing" paper. It is a self-contained
example that uses mock objects for models and tokenizers to focus purely
on the template's workflow without requiring actual model loading.

Workflow:
1.  **Plan Stage (Large Model)**: A mock "large model" generates a predefined
    step-by-step plan for a given math problem.
2.  **Solve Stage (Small Model)**: A mock "small model" takes the problem
    and the generated plan, then produces a predefined solution.
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
        # Simulate model generation based on the prompt type (plan vs solve)
        # The actual token IDs don't matter here, as the mock tokenizer will decode them.
        if "Plan:" in MOCK_TOKENIZER.decode(input_ids[0]):
            # This is a planning prompt for the large model
            return [[0] * 20] # Return a fixed-length list of mock token IDs
        else:
            # This is a solving prompt for the small model
            return [[1] * 50] # Return a different list of mock token IDs
            
    def to(self, device):
        # Mock device placement
        return self

class MockTokenizer:
    """A mock class to simulate a Hugging Face tokenizer."""
    def __call__(self, text_list, return_tensors="pt"):
        # Simulate tokenization
        return {"input_ids": [[i for i, char in enumerate(text_list[0])]]}

    def encode(self, text, add_special_tokens=False):
        # Simulate encoding
        return [123]

    def decode(self, token_ids, skip_special_tokens=True):
        # Simulate decoding based on the mock token IDs from the model
        if token_ids and token_ids[0] == 0:
            return "Step 1: Identify the key numbers. Step 2: Perform the main calculation. <end>"
        elif token_ids and token_ids[0] == 1:
            return "First, we take 10 and multiply by 5, which gives 50. Then we add 3. The final answer is 53. #### 53 <end>"
        return "".join([str(t) for t in token_ids])

# Instantiate the mock tokenizer globally for the mock model to use
MOCK_TOKENIZER = MockTokenizer()

# --- 1. 配置模块 (Configuration Module) ---
class Config:
    """Minimal configuration for the template."""
    DEVICE: str = "cpu"
    PLAN_MAX_NEW_TOKENS: int = 128
    SOLVE_MAX_NEW_TOKENS: int = 350
    TEMPERATURE: float = 0.1
    TOP_P: float = 0.9
    DO_SAMPLE: bool = False
    FEWSHOT_PROMPT_PATH: str = "non_existent_path.txt" # Path not needed for this example


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
        
        # The mock model's generate method is simplified
        generated_ids = model.generate(inputs['input_ids'])

        # The mock tokenizer's decode method handles the logic
        response_full = tokenizer.decode(generated_ids[0])
        response = response_full.split(stop_sequence)[0].strip()

        # Simulate token count
        num_tokens = len(response.split())
        return response, num_tokens

    def run(self, query: str) -> Tuple[str, int, int]:
        """This method should be implemented by subclasses."""
        raise NotImplementedError

class PlanAndSolveTemplate(BaseTemplate):
    """
    Implements the Plan-and-Solve (PS) domain-specific template for GSM8K.
    """
    def run(self, query: str) -> Tuple[str, int, int]:
        """Executes the Plan-and-Solve workflow."""
        print("--- Executing Plan-and-Solve Template ---")
        
        # --- Stage 1: Plan (Large Model) ---
        print("\n[Stage 1] Generating plan with Large Model...")
        plan_prompt = (
            f"Question: {query}\n\n"
            "Provide a step-by-step plan to solve this problem. Do not write the final answer.\n"
            "Plan:"
        )
        plan, large_model_tokens = self._generate(
            model=self.mm.large_model,
            tokenizer=self.mm.large_tokenizer,
            prompt=plan_prompt,
            max_new_tokens=self.config.PLAN_MAX_NEW_TOKENS
        )
        print(f"Generated Plan: '{plan}'")

        # --- Stage 2: Solve (Small Model) ---
        print("\n[Stage 2] Generating solution with Small Model...")
        solve_prompt = (
            f"Based on the following plan, solve the math problem.\n\n"
            f"Question: {query}\n\n"
            f"Plan:\n{plan}\n\n"
            "Now, provide the detailed solution and the final answer.\n"
            "Solution:"
        )
        solution, small_model_tokens = self._generate(
            model=self.mm.small_model,
            tokenizer=self.mm.small_tokenizer,
            prompt=solve_prompt,
            max_new_tokens=self.config.SOLVE_MAX_NEW_TOKENS
        )
        print(f"Generated Solution: '{solution}'")

        total_response = f"--- Plan (Generated by Large Model) ---\n{plan}\n\n--- Solution (Generated by Small Model) ---\n{solution}"
        return total_response, large_model_tokens, small_model_tokens


# --- 4. 主执行函数 (Main Execution Function) ---
def main():
    """Main function to demonstrate the PlanAndSolveTemplate."""
    print("🚀 Initializing standalone Plan-and-Solve demo...")
    
    # Configuration and mock manager setup
    config = Config()
    mock_manager = MockModelManager(config)
    
    # Instantiate the template
    ps_template = PlanAndSolveTemplate(mock_manager, config)
    
    # Define a sample query
    sample_query = "If a train travels at 10 miles per hour for 5 hours and then stops for a snack that costs $3, what is the total distance traveled?"
    print(f"\nSample Query: \"{sample_query}\"")
    
    # Run the template
    final_response, plan_tokens, solve_tokens = ps_template.run(sample_query)
    
    # Print the final results
    print("\n--- ✅ Final Combined Output ---")
    print(final_response)
    print("\n--- 📊 Summary ---")
    print(f"Tokens from Large Model (Plan): {plan_tokens}")
    print(f"Tokens from Small Model (Solve): {solve_tokens}")
    print(f"Total Generated Tokens: {plan_tokens + solve_tokens}")

if __name__ == "__main__":
    main()
