# main.py

"""
Main execution script for the KNN Difficulty Classifier.

This script orchestrates the entire process:
1. Initializes the KNN classifier.
2. Fits the classifier with the reference dataset.
3. Predicts the difficulty for the candidate dataset.
4. Saves the results to a new .jsonl file.
"""

from knn_classifier import KnnDifficultyClassifier
from config import (
    REFERENCE_DATA_PATH,
    CANDIDATE_DATA_PATH,
    OUTPUT_PATH,
    K_NEIGHBORS,
    EASY_DIFFICULTY_THRESHOLD
)

def main() -> None:
    """
    Main function to run the difficulty classification process.
    """
    # Step 1: Initialize the classifier
    classifier = KnnDifficultyClassifier()
    
    # Step 2: Fit the classifier with the reference data
    classifier.fit(reference_data_path=REFERENCE_DATA_PATH)
    
    # Step 3: Predict the difficulty for the candidate data
    results = classifier.predict(
        candidate_data_path=CANDIDATE_DATA_PATH,
        k=K_NEIGHBORS,
        easy_threshold=EASY_DIFFICULTY_THRESHOLD
    )
    
    # Step 4: Save the final results
    classifier.save_results(data=results, output_path=OUTPUT_PATH)

if __name__ == "__main__":
    main()