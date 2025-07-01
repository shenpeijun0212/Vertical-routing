from modelscope import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch
import datetime
import time
import json
import re
import os
import numpy as np
import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.tokenize import word_tokenize
import jieba

# Configuration Section
# Model paths
LARGE_MODEL_PATH = "/gemini/code/model/Qwen/Qwen2.5-7B"
SMALL_MODEL_PATH = "/gemini/code/model/Qwen/Qwen2.5-3B"

# Few-shot prompt file path
FEWSHOT_PROMPT_PATH = "/gemini/code/RouteAgent/prompt/MT_8-shot.txt"

# Test data path
TEST_DATA_PATH = "/gemini/code/RouteAgent/data/WMT_translation/test.jsonl"

# Data size (number of samples to evaluate)
DATA_SIZE = 100

# Initialize Paths and Models
# Get current date and time for result file naming
now = datetime.now()
current_date = now.strftime('%Y-%m-%d')  # Format: 'YYYY-MM-DD'
current_hour = now.strftime('%H')        # Format: 'HH'

# Create result directory structure
RESULT_SAVE_BASE_PATH = "/gemini/code/RouteAgent/result"
DATE_FOLDER_PATH = os.path.join(RESULT_SAVE_BASE_PATH, current_date)
os.makedirs(DATE_FOLDER_PATH, exist_ok=True)

# Construct result file path
RESULT_FILE_PATH = os.path.join(
    DATE_FOLDER_PATH,
    f"{current_hour}_{os.path.basename(LARGE_MODEL_PATH)}_{os.path.basename(SMALL_MODEL_PATH)}_GSM8K_{DATA_SIZE}_25%.json"
)

# Update result save path (Note: This seems redundant as we're writing directly to RESULT_FILE_PATH)
result_save_path = RESULT_FILE_PATH
device = "cuda"  # Using GPU for inference

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

# Helper Functions
def tokenize_chinese(text):
    """
    Tokenize Chinese text using jieba.
    
    Args:
        text (str): Chinese text to tokenize
        
    Returns:
        list: List of tokens
    """
    return list(jieba.cut(text))

# Main Evaluation Loop
# Initialize result storage
ACC_RES = []          # Accumulated responses
REFERENCES = []       # Ground truth translations (Chinese)
CANDIDATES = []       # Model translations (Chinese)

# Start total timer
TOTAL_START_TIME = time.time()

# Download NLTK resources (only needed once)
nltk.download('punkt', quiet=True)

# Process each test case
with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
    cnt = 0
    for line in f:
        # Parse JSON line
        i = json.loads(line.strip())
        REFERENCES.append(i['zh'])  # Store reference translation
        cnt += 1
        each_start_time = time.time()
        
        # Large Model Inference    
        # Construct prompt for large model
        LARGE_PROMPT = FEWSHOT_PROMPT + i['en'] + "\n" + "Keywords:"
        print(i['en'])  # Print source sentence
        print("--------Translation--------")
        
        # Tokenize and generate response
        LARGE_MODEL_INPUTS = LARGE_TOKENIZER([LARGE_PROMPT], return_tensors="pt").to(device)
        LARGE_STOP_SEQUENCE = "####"
        LARGE_STOP_TOKEN_IDS = LARGE_TOKENIZER.encode(LARGE_STOP_SEQUENCE, add_special_tokens=False)
        LARGE_EOS_TOKEN_ID = LARGE_STOP_TOKEN_IDS[0]
        
        LARGE_GENERATED_IDS = LARGE_MODEL.generate(
            LARGE_MODEL_INPUTS.input_ids,
            max_new_tokens=256,
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
        
        # Decode response
        LARGE_RESPONSE = LARGE_TOKENIZER.batch_decode(LARGE_GENERATED_IDS, skip_special_tokens=True)[0]
        
        # Small Model Inference
        # Construct prompt for small model (using large model's response)
        SMALL_PROMPT = LARGE_PROMPT + LARGE_RESPONSE + "\n" + "Sentence Translation: "
        
        # Extract large model response before stop sequence
        LARGE_RESPONSE = LARGE_RESPONSE.split(LARGE_STOP_SEQUENCE)[0]
        
        # Tokenize and generate response
        SMALL_MODEL_INPUTS = SMALL_TOKENIZER([SMALL_PROMPT], return_tensors="pt").to(device)
        SMALL_STOP_SEQUENCE = "<end>"
        SMALL_STOP_TOKEN_IDS = SMALL_TOKENIZER.encode(SMALL_STOP_SEQUENCE, add_special_tokens=False)
        SMALL_EOS_TOKEN_ID = SMALL_STOP_TOKEN_IDS[0]
        
        SMALL_GENERATED_IDS = SMALL_MODEL.generate(
            SMALL_MODEL_INPUTS.input_ids,
            max_new_tokens=256,
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
        
        # Decode response and clean up
        SMALL_RESPONSE = SMALL_TOKENIZER.batch_decode(SMALL_GENERATED_IDS, skip_special_tokens=True)[0]
        SMALL_RESPONSE = SMALL_RESPONSE.split(SMALL_STOP_SEQUENCE)[0]
        SMALL_RESPONSE = SMALL_RESPONSE.split("<")[0]
        
        # Combine responses
        response = LARGE_RESPONSE + SMALL_RESPONSE
        
        # Record Results
        each_end_time = time.time()
        elapsed_time = each_end_time - each_start_time
        current_time = datetime.now()
        
        # Store results
        ACC_RES.append(SMALL_RESPONSE)
        CANDIDATES.append(SMALL_RESPONSE.rstrip())
        
        # Log processing information
        text_to_write = (
            f"Case: {cnt} | Current time: {current_time} | "
            f"Elapsed Time: {elapsed_time:.2f} seconds"
        )
        print(text_to_write)
        print(SMALL_RESPONSE)  # Print final translation
        

# Calculate BLEU Scores
# Tokenize references and candidates
references_tokenized = [tokenize_chinese(ref) for ref in REFERENCES]
candidates_tokenized = [tokenize_chinese(cand) for cand in CANDIDATES]

# Print tokenization results (optional)
# print("References Tokenized:", references_tokenized)
# print("Candidates Tokenized:", candidates_tokenized)

# Calculate and print BLEU scores for each translation
for i, (ref, cand) in enumerate(zip(references_tokenized, candidates_tokenized)):
    bleu_1 = sentence_bleu([ref], cand, weights=(1, 0, 0, 0))  # BLEU-1
    bleu_2 = sentence_bleu([ref], cand, weights=(0.5, 0.5, 0, 0))  # BLEU-2
    bleu_3 = sentence_bleu([ref], cand, weights=(0.33, 0.33, 0.33, 0))  # BLEU-3
    bleu_4 = sentence_bleu([ref], cand, weights=(0.25, 0.25, 0.25, 0.25))  # BLEU-4
    print(f"Translation {i+1}:")
    print(f"  BLEU-1: {bleu_1:.4f}")
    print(f"  BLEU-2: {bleu_2:.4f}")
    print(f"  BLEU-3: {bleu_3:.4f}")
    print(f"  BLEU-4: {bleu_4:.4f}")
    print("-" * 40)

# Calculate overall corpus BLEU scores
def calculate_corpus_bleu(references, candidates, weights):
    """
    Calculate average BLEU score for the entire corpus.
    
    Args:
        references (list): List of tokenized reference translations
        candidates (list): List of tokenized candidate translations
        weights (tuple): Weights for n-gram precision
        
    Returns:
        float: Average BLEU score
    """
    return sum(
        sentence_bleu([ref], cand, weights=weights)
        for ref, cand in zip(references, candidates)
    ) / len(candidates)

corpus_bleu_1 = calculate_corpus_bleu(references_tokenized, candidates_tokenized, weights=(1, 0, 0, 0))
corpus_bleu_2 = calculate_corpus_bleu(references_tokenized, candidates_tokenized, weights=(0.5, 0.5, 0, 0))
corpus_bleu_3 = calculate_corpus_bleu(references_tokenized, candidates_tokenized, weights=(0.33, 0.33, 0.33, 0))
corpus_bleu_4 = calculate_corpus_bleu(references_tokenized, candidates_tokenized, weights=(0.25, 0.25, 0.25, 0.25))

print("Corpus Average Scores:")
print(f"  BLEU-1: {corpus_bleu_1:.4f}")
print(f"  BLEU-2: {corpus_bleu_2:.4f}")
print(f"  BLEU-3: {corpus_bleu_3:.4f}")
print(f"  BLEU-4: {corpus_bleu_4:.4f}")

# Save Results
# Save all responses to JSONL file
with open(result_save_path, "w", encoding="utf-8") as f:
    for item in CANDIDATES:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

print(f"\nResults successfully saved to: {result_save_path}")