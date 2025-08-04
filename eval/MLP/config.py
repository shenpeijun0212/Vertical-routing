# config.py

"""
Configuration file for the MLP Difficulty Classifier project.

This file contains file paths, model architecture, and training
hyperparameters to make them easily accessible and modifiable.
"""
import torch

# --- Device Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Path Configuration ---
# Path to the pre-trained SentenceTransformer model for feature extraction
EMBEDDING_MODEL_PATH = ""

# Path to the full dataset with labels, to be split for training and validation
# Note: For a robust workflow, you should split this into explicit train/validation files.
# For this example, we'll split it programmatically.
LABELED_DATA_PATH = ""

# Path to the candidate dataset (unlabeled data for prediction)
PREDICTION_DATA_PATH = ""

# Path to save the trained MLP model
MODEL_SAVE_PATH = ""

# Path for the final prediction output file
PREDICTION_OUTPUT_PATH = ""


# --- Model Architecture ---
# The dimension of the input embeddings from the SentenceTransformer model.
# Most base models use 768, large models might use 1024. Please verify for your model.
INPUT_DIM = 768
HIDDEN_DIM = 128
OUTPUT_DIM = 2  # Two classes: 0 for 'difficult', 1 for 'easy'

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
EPOCHS = 20
# Ratio for splitting the labeled data into training and validation sets
VALIDATION_SPLIT = 0.2