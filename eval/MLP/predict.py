# predict.py
"""
Script for making predictions on new data using a trained MLP classifier.

This script performs the following steps:
1. Loads the feature extractor and the trained MLP model weights.
2. Loads the unlabeled candidate data.
3. Generates embeddings for the candidate data.
4. Makes predictions using the trained model.
5. Saves the data with the added predictions to a new file.
"""
import torch
import json
from typing import List, Dict, Any
from tqdm import tqdm

import config
from mlp_model import MLP
from data_loader import load_jsonl
from sentence_transformers import SentenceTransformer


def make_predictions():
    """Orchestrates the prediction process."""
    print(f"Using device: {config.DEVICE}")
    
    # 1. Load models
    feature_extractor = SentenceTransformer(config.EMBEDDING_MODEL_PATH, device=config.DEVICE)
    
    model = MLP(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=config.OUTPUT_DIM
    ).to(config.DEVICE)
    
    print(f"Loading trained model weights from {config.MODEL_SAVE_PATH}")
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    model.eval()

    # 2. Load candidate data
    candidate_data = load_jsonl(config.PREDICTION_DATA_PATH)
    queries = [item['query'] for item in candidate_data]
    
    # 3. Generate embeddings
    print("Generating embeddings for prediction data...")
    embeddings = feature_extractor.encode(
        queries,
        convert_to_tensor=True,
        show_progress_bar=True
    ).to(config.DEVICE)

    # 4. Make predictions
    print("Making predictions...")
    results_with_difficulty: List[Dict[str, Any]] = []
    
    # Define the reverse mapping from integer to label
    label_map_rev = {0: 'difficult', 1: 'easy'}

    with torch.no_grad():
        for i in tqdm(range(len(embeddings)), desc="Predicting"):
            single_embedding = embeddings[i].unsqueeze(0) # Add batch dimension
            output = model(single_embedding)
            prediction_idx = torch.argmax(output, dim=1).item()
            
            # Add the prediction to the original data item
            data_item = candidate_data[i]
            data_item['dif'] = label_map_rev[prediction_idx]
            results_with_difficulty.append(data_item)

    # 5. Save results
    with open(config.PREDICTION_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for item in results_with_difficulty:
            f.write(json.dumps(item) + '\n')
            
    print(f"\nPredictions saved successfully to {config.PREDICTION_OUTPUT_PATH}")


if __name__ == "__main__":
    make_predictions()