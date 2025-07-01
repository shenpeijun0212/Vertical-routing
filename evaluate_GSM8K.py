from modelscope import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch
import json
import os
import time
import re
import numpy as np
from datetime import datetime

# Configuration Section
# Model paths
LARGE_MODEL_PATH = "/gemini/code/model/Qwen/Qwen2.5-32B"
SMALL_MODEL_PATH = "/gemini/code/model/Qwen/Qwen2.5-7B"

# Few-shot prompt file path
FEWSHOT_PROMPT_PATH = "/gemini/code/RouteAgent/MATH/base_8-shot.txt"

# Test data path
TEST_DATA_PATH = "/gemini/code/RouteAgent/data/GSM8K_500.jsonl"

# Data size (number of samples to evaluate)
DATA_SIZE = 500

# Initialize Paths and Models
# Get current date and time for result file naming
now = datetime.now()
current_date = now.strftime('%Y-%m-%d')  # Format: 'YYYY-MM-DD'
current_hour = now.strftime('%H')        # Format: 'HH'

# Create result directory structure
RESULT_SAVE_BASE_PATH = ""
DATE_FOLDER_PATH = os.path.join(RESULT_SAVE_BASE_PATH, current_date)
os.makedirs(DATE_FOLDER_PATH, exist_ok=True)

# Construct result file path
RESULT_FILE_PATH = os.path.join(
    DATE_FOLDER_PATH,
    f"{current_hour}_{os.path.basename(LARGE_MODEL_PATH)}_{os.path.basename(SMALL_MODEL_PATH)}_GSM8K_{DATA_SIZE}_50%.json"
)

# Update result save path (Note: This seems redundant as we're writing directly to RESULT_FILE_PATH)
result_save_path = RESULT_FILE_PATH
print(f"Result will be saved to: {result_save_path}")

# Device configuration
DEVICE = "cuda"  # Using GPU for inference

# Load models
LARGE_MODEL = AutoModelForCausalLM.from_pretrained(
    LARGE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

SMALL_MODEL = AutoModelForCausalLM.from_pretrained(
    SMALL_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Load tokenizers
LARGE_TOKENIZER = AutoTokenizer.from_pretrained(LARGE_MODEL_PATH)
SMALL_TOKENIZER = AutoTokenizer.from_pretrained(SMALL_MODEL_PATH)

# ======================
# Helper Functions
# ======================

def extract_number_after_hash(text):
    """
    Extract the first number following '####' in the given text.
    
    Args:
        text (str): Input text containing the answer pattern
        
    Returns:
        str: Extracted number or None if not found
    """
    match = re.search(r'####\s*(\d+)', text)
    return match.group(1) if match else None

# Stopping token configuration
STOP_TOKEN_LIMIT = 90


# Main Evaluation Loop
# Initialize result storage
ACC_RES = []          # Accumulated responses
REFERENCES = []       # Ground truth answers
CANDIDATES = []       # Model predictions

# Start total timer
TOTAL_START_TIME = time.time()

# Process each test case
with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
    cnt = 0
    for line in f:
        # Parse JSON line
        i = json.loads(line.strip())
        cnt += 1
        each_start_time = time.time()
        
        # Large Model Inference
        # Construct prompt for large model
        LARGE_PROMPT = FEWSHOT_PROMPT + i['query'] + "\n"
        
        # Tokenize and generate response
        LARGE_MODEL_INPUTS = LARGE_TOKENIZER([LARGE_PROMPT], return_tensors="pt").to(DEVICE)
        LARGE_STOP_SEQUENCE = "end"
        LARGE_STOP_TOKEN_IDS = LARGE_TOKENIZER.encode(LARGE_STOP_SEQUENCE, add_special_tokens=False)
        LARGE_EOS_TOKEN_ID = LARGE_STOP_TOKEN_IDS[0]
        
        LARGE_GENERATED_IDS = LARGE_MODEL.generate(
            LARGE_MODEL_INPUTS.input_ids,
            max_new_tokens=STOP_TOKEN_LIMIT,
            do_sample=False,
            temperature=0.1,
            eos_token_id=LARGE_EOS_TOKEN_ID,
            top_p=0.9,
        )
        
        # Process generated tokens
        LARGE_GENERATED_IDS = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(LARGE_MODEL_INPUTS.input_ids, LARGE_GENERATED_IDS)
        ]
        LARGE_OUTPUT_TOKENS = len(LARGE_GENERATED_IDS[0])
        
        # Decode response
        LARGE_RESPONSE = LARGE_TOKENIZER.batch_decode(LARGE_GENERATED_IDS, skip_special_tokens=True)[0]
        
        # Small Model Inference
        # Construct prompt for small model (using large model's response)
        SMALL_PROMPT = LARGE_PROMPT + LARGE_RESPONSE
        
        # Tokenize and generate response
        SMALL_MODEL_INPUTS = SMALL_TOKENIZER([SMALL_PROMPT], return_tensors="pt").to(DEVICE)
        SMALL_STOP_SEQUENCE = "end"
        SMALL_STOP_TOKEN_IDS = SMALL_TOKENIZER.encode(SMALL_STOP_SEQUENCE, add_special_tokens=False)
        SMALL_EOS_TOKEN_ID = SMALL_STOP_TOKEN_IDS[0]
        
        SMALL_GENERATED_IDS = SMALL_MODEL.generate(
            SMALL_MODEL_INPUTS.input_ids,
            max_new_tokens=350,
            do_sample=False,
            temperature=0.1,
            eos_token_id=SMALL_EOS_TOKEN_ID,
            top_p=0.9,
        )
        
        # Process generated tokens
        SMALL_GENERATED_IDS = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(SMALL_MODEL_INPUTS.input_ids, SMALL_GENERATED_IDS)
        ]
        SMALL_OUTPUT_TOKENS = len(SMALL_GENERATED_IDS[0])
        
        # Decode response and clean up
        SMALL_RESPONSE = SMALL_TOKENIZER.batch_decode(SMALL_GENERATED_IDS, skip_special_tokens=True)[0]
        SMALL_RESPONSE = SMALL_RESPONSE.split("<end")[0]  # Remove any trailing tags
        
        # Combine responses
        TOTAL_RESPONSE = LARGE_RESPONSE + SMALL_RESPONSE

        # Record Results        
        each_end_time = time.time()
        elapsed_time = each_end_time - each_start_time
        current_time = datetime.now()
        
        # Store results
        ACC_RES.append(TOTAL_RESPONSE)
        REFERENCE = extract_number_after_hash(i['response'])
        REFERENCES.append(REFERENCE)
        CANDIDATE = extract_number_after_hash(TOTAL_RESPONSE)
        CANDIDATES.append(CANDIDATE)
        
        # Log processing information
        output_tokens = LARGE_OUTPUT_TOKENS + SMALL_OUTPUT_TOKENS
        log_message = (
            f"Case: {cnt} | Current time: {current_time} | "
            f"Elapsed Time: {elapsed_time:.2f}s | "
            f"Total Tokens: {output_tokens}"
        )
        print(log_message)
        
# Calculate Metrics
# Calculate accuracy
correct_count = sum(1 for ref, cand in zip(REFERENCES, CANDIDATES) if ref == cand)
wrong_count = len(REFERENCES) - correct_count
total_count = correct_count + wrong_count
accuracy = correct_count / total_count if total_count > 0 else 0

# Print evaluation results
print("\nEvaluation Results:")
print(f"Total Questions: {total_count}")
print(f"Correct Answers: {correct_count}")
print(f"Wrong Answers: {wrong_count}")
print(f"Accuracy: {accuracy:.4f}")

# Save Results
# Save all responses to JSONL file
with open(result_save_path, "w", encoding="utf-8") as f:
    for item in ACC_RES:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

print(f"\nResults successfully saved to: {result_save_path}")