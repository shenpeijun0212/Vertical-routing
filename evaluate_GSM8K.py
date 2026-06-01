# -*- coding: utf-8 -*-
"""
Vertical Routing for GSM8K Evaluation

This script implements the "Vertical Routing" framework described in the paper
"Vertical Routing: A Cost-Efficient Collaboration Routing Framework".
Specifically, it uses a Domain-Specific Template tailored for mathematical
reasoning tasks like GSM8K: the "Plan-and-Solve" (PS) template.

Methodology:
1.  **Plan Stage (High Criticality)**: A powerful Large Language Model (LLM)
    is prompted to generate a high-level, step-by-step plan to solve the
    mathematical problem. This is the most critical stage, as a correct
    plan is essential for a correct solution.
2.  **Solve Stage (Low Criticality)**: A smaller, more cost-efficient LLM
    takes the original problem and the generated plan as input, and executes
    the calculations to arrive at the final answer.

This approach allocates computational resources intelligently, using the
large model for critical reasoning (planning) and the small model for
more straightforward execution (solving), optimizing for both accuracy
and cost.
"""
import json
import os
import re
import time
from datetime import datetime
from typing import Dict, Any, Tuple, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

# --- 1. 配置模块 (Configuration Module) ---
class Config:
    """
    Centralized configuration for the evaluation script.
    """
    # Model Paths
    LARGE_MODEL_PATH: str = "/gemini/code/model/Qwen/Qwen2.5-32B"
    SMALL_MODEL_PATH: str = "/gemini/code/model/Qwen/Qwen2.5-7B"

    # Data and Prompt Paths
    FEWSHOT_PROMPT_PATH: str = "/gemini/code/RouteAgent/MATH/base_8-shot.txt"
    TEST_DATA_PATH: str = "/gemini/code/RouteAgent/data/GSM8K_500.jsonl"
    DATA_SIZE: int = 500 # Number of samples to evaluate

    # Device Configuration
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Generation Parameters
    # Parameters for the planning stage (Large Model)
    PLAN_MAX_NEW_TOKENS: int = 128
    PLAN_TEMPERATURE: float = 0.1
    PLAN_TOP_P: float = 0.9
    PLAN_DO_SAMPLE: bool = False

    # Parameters for the solving stage (Small Model)
    SOLVE_MAX_NEW_TOKENS: int = 350
    SOLVE_TEMPERATURE: float = 0.1
    SOLVE_TOP_P: float = 0.9
    SOLVE_DO_SAMPLE: bool = False

    # Result Saving
    RESULT_SAVE_BASE_PATH: str = "./results"


# --- 2. 模型管理模块 (Model Management Module) ---
class ModelManager:
    """
    Handles loading and managing LLMs and their tokenizers.
    """
    def __init__(self, config: Config):
        self.config = config
        self.large_model, self.large_tokenizer = self._load_model(config.LARGE_MODEL_PATH)
        self.small_model, self.small_tokenizer = self._load_model(config.SMALL_MODEL_PATH)
        print("✅ Models and tokenizers loaded successfully.")

    def _load_model(self, model_path: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Loads a model and its tokenizer from a given path."""
        print(f"Loading model from: {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer

# --- 3. 模板定义模块 (Template Definition Module) ---
class PlanAndSolveTemplate:
    """
    Implements the Plan-and-Solve (PS) domain-specific template for GSM8K.
    """
    def __init__(self, model_manager: ModelManager, config: Config):
        self.mm = model_manager
        self.config = config
        self.fewshot_prompt = self._load_fewshot_prompt()

    def _load_fewshot_prompt(self) -> str:
        """Loads the few-shot examples from a file."""
        try:
            with open(self.config.FEWSHOT_PROMPT_PATH, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Warning: Few-shot prompt file not found at {self.config.FEWSHOT_PROMPT_PATH}. Using empty prompt.")
            return ""

    def _generate(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float, do_sample: bool) -> Tuple[str, int]:
        """Generic text generation function."""
        inputs = tokenizer([prompt], return_tensors="pt").to(self.config.DEVICE)
        
        # Define a stopping sequence. The model should stop generating after this.
        stop_sequence = "end"
        stop_token_ids = tokenizer.encode(stop_sequence, add_special_tokens=False)
        eos_token_id = stop_token_ids[0] if stop_token_ids else tokenizer.eos_token_id

        generated_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
        )

        # Extract only the newly generated tokens
        output_ids = generated_ids[0][len(inputs.input_ids[0]):]
        response = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        # Clean up the response
        response = response.split("<end")[0].strip()

        return response, len(output_ids)

    def run(self, query: str) -> Tuple[str, int, int]:
        """
        Executes the Plan-and-Solve workflow.

        Returns:
            A tuple containing:
            - The final combined response (plan + solution).
            - The number of tokens generated by the large model.
            - The number of tokens generated by the small model.
        """
        # --- Stage 1: Plan (Large Model) ---
        # Create a prompt specifically for planning
        plan_prompt = (
            f"{self.fewshot_prompt.strip()}\n\n"
            f"Question: {query}\n\n"
            "Provide a step-by-step plan to solve this problem. Do not write the final answer.\n"
            "Plan:"
        )
        
        plan, large_model_tokens = self._generate(
            model=self.mm.large_model,
            tokenizer=self.mm.large_tokenizer,
            prompt=plan_prompt,
            max_new_tokens=self.config.PLAN_MAX_NEW_TOKENS,
            temperature=self.config.PLAN_TEMPERATURE,
            top_p=self.config.PLAN_TOP_P,
            do_sample=self.config.PLAN_DO_SAMPLE
        )

        # --- Stage 2: Solve (Small Model) ---
        # Create a prompt for solving, incorporating the generated plan
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
            max_new_tokens=self.config.SOLVE_MAX_NEW_TOKENS,
            temperature=self.config.SOLVE_TEMPERATURE,
            top_p=self.config.SOLVE_TOP_P,
            do_sample=self.config.SOLVE_DO_SAMPLE
        )

        # Combine the plan and solution for a full, interpretable response
        total_response = f"--- Plan (Generated by Large Model) ---\n{plan}\n\n--- Solution (Generated by Small Model) ---\n{solution}"
        
        return total_response, large_model_tokens, small_model_tokens


# --- 4. 评估器模块 (Evaluator Module) ---
class Evaluator:
    """
    Orchestrates the entire evaluation process.
    """
    def __init__(self, config: Config, model_manager: ModelManager, template: PlanAndSolveTemplate):
        self.config = config
        self.model_manager = model_manager
        self.template = template
        self.results: List[Dict[str, Any]] = []
        self.result_file_path = self._setup_result_path()

    def _setup_result_path(self) -> str:
        """Creates directory and file path for saving results."""
        now = datetime.now()
        date_folder = now.strftime('%Y-%m-%d')
        hour_str = now.strftime('%H')
        
        # Create a descriptive filename
        large_model_name = os.path.basename(self.config.LARGE_MODEL_PATH)
        small_model_name = os.path.basename(self.config.SMALL_MODEL_PATH)
        filename = f"{hour_str}H_{large_model_name}_vs_{small_model_name}_GSM8K_PS-Template_{self.config.DATA_SIZE}.jsonl"
        
        # Create directory if it doesn't exist
        full_dir_path = os.path.join(self.config.RESULT_SAVE_BASE_PATH, date_folder)
        os.makedirs(full_dir_path, exist_ok=True)
        
        return os.path.join(full_dir_path, filename)

    @staticmethod
    def _extract_answer(text: str) -> str:
        """Extracts the final numerical answer from the text."""
        # The pattern looks for the last number in the format '#### 123'
        matches = re.findall(r'####\s*(\d+\.?\d*)', text)
        return matches[-1] if matches else None

    def run(self):
        """Runs the evaluation loop across the dataset."""
        print(f"🚀 Starting evaluation on {self.config.TEST_DATA_PATH}...")
        total_start_time = time.time()
        
        with open(self.config.TEST_DATA_PATH, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= self.config.DATA_SIZE:
                    break
                
                data_item = json.loads(line.strip())
                case_start_time = time.time()

                # Use the template to get the response
                response, large_tokens, small_tokens = self.template.run(data_item['question'])

                case_end_time = time.time()
                elapsed_time = case_end_time - case_start_time
                total_tokens = large_tokens + small_tokens
                
                # Extract reference and candidate answers
                reference_answer = self._extract_answer(data_item['answer'])
                candidate_answer = self._extract_answer(response)
                
                is_correct = reference_answer == candidate_answer

                # Store detailed results for analysis
                self.results.append({
                    'case_id': i + 1,
                    'question': data_item['question'],
                    'reference_answer_raw': data_item['answer'],
                    'reference_answer_extracted': reference_answer,
                    'candidate_answer_raw': response,
                    'candidate_answer_extracted': candidate_answer,
                    'is_correct': is_correct,
                    'elapsed_time_s': elapsed_time,
                    'large_model_tokens': large_tokens,
                    'small_model_tokens': small_tokens,
                    'total_tokens': total_tokens,
                })

                print(
                    f"Case: {i+1}/{self.config.DATA_SIZE} | "
                    f"Correct: {is_correct} | "
                    f"Ref: {reference_answer} | Cand: {candidate_answer} | "
                    f"Time: {elapsed_time:.2f}s | "
                    f"Tokens (L/S/T): {large_tokens}/{small_tokens}/{total_tokens}"
                )

        self._finalize_and_save()
        total_elapsed = time.time() - total_start_time
        print(f"\nTotal evaluation time: {total_elapsed:.2f} seconds.")

    def _finalize_and_save(self):
        """Calculates final metrics and saves results to a file."""
        if not self.results:
            print("No results to analyze.")
            return

        correct_count = sum(1 for r in self.results if r['is_correct'])
        total_count = len(self.results)
        accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0

        avg_time = sum(r['elapsed_time_s'] for r in self.results) / total_count
        avg_tokens = sum(r['total_tokens'] for r in self.results) / total_count

        summary = {
            'total_questions': total_count,
            'correct_answers': correct_count,
            'wrong_answers': total_count - correct_count,
            'accuracy_percent': accuracy,
            'average_time_per_case_s': avg_time,
            'average_tokens_per_case': avg_tokens,
            'model_large': self.config.LARGE_MODEL_PATH,
            'model_small': self.config.SMALL_MODEL_PATH,
            'template': 'PlanAndSolve',
        }

        print("\n--- 📊 Evaluation Summary ---")
        for key, value in summary.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        print("----------------------------")

        # Save detailed results to a JSONL file
        with open(self.result_file_path, "w", encoding="utf-8") as f:
            # Write summary as the first line (optional, but good for context)
            f.write(json.dumps({'summary': summary}) + '\n')
            # Write individual case results
            for result in self.results:
                f.write(json.dumps(result) + '\n')

        print(f"\n✅ Results successfully saved to: {self.result_file_path}")


# --- 5. 主执行函数 (Main Execution Function) ---
def main():
    """Main function to run the script."""
    try:
        config = Config()
        model_manager = ModelManager(config)
        template = PlanAndSolveTemplate(model_manager, config)
        evaluator = Evaluator(config, model_manager, template)
        evaluator.run()
    except Exception as e:
        print(f"An error occurred: {e}")
        # Add more specific error handling if needed
        # For example, for torch.cuda.OutOfMemoryError
        if isinstance(e, torch.cuda.OutOfMemoryError):
            print("CUDA out of memory. Try reducing batch size or using a smaller model.")

if __name__ == "__main__":
    main()
