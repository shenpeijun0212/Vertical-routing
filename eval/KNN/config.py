# config.py

"""
Configuration file for the KNN Difficulty Classifier project.

This file contains all the file paths and model parameters
to make them easily accessible and modifiable without changing
the core logic of the application.
"""

# Path to the pre-trained SentenceTransformer model
EMBEDDING_MODEL_PATH = ""

# --- File Paths ---
# Path to the reference dataset (the larger dataset with known difficulties)
REFERENCE_DATA_PATH = ""

# Path to the candidate dataset (the dataset to be classified)
CANDIDATE_DATA_PATH = ""

# Path for the final output file with difficulty predictions
OUTPUT_PATH = ""


# --- Model Parameters ---
# The number of nearest neighbors to consider for classification
K_NEIGHBORS = 10

# The threshold for classifying a problem as 'easy'. If the number of 'easy'
# neighbors is greater than this value, the problem is classified as 'easy'.
# For instance, if K_NEIGHBORS is 5 and this threshold is 4, it means
# all 5 neighbors must be 'easy' for the candidate to be 'easy'.
EASY_DIFFICULTY_THRESHOLD = 4