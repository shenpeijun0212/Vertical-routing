# knn_classifier.py

"""
Core logic for the K-Nearest Neighbors (KNN) based difficulty classifier.

This module defines the KnnDifficultyClassifier class, which handles
loading data, generating sentence embeddings, and performing the
classification of problem difficulty based on the nearest neighbors
in a reference dataset.
"""

import json
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_PATH

class KnnDifficultyClassifier:
    """
    A classifier to determine the difficulty of problems using a KNN approach.
    """

    def __init__(self, model_path: str = EMBEDDING_MODEL_PATH):
        """
        Initializes the classifier by loading the SentenceTransformer model.

        Args:
            model_path (str): The path to the pre-trained SentenceTransformer model.
        """
        print("Loading the sentence embedding model...")
        self.model = SentenceTransformer(model_path)
        self.reference_embeddings = None
        self.reference_data = None

    def _load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Loads a .jsonl file into a list of dictionaries.

        Args:
            file_path (str): The path to the .jsonl file.

        Returns:
            List[Dict[str, Any]]: A list where each element is a JSON object from the file.
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def fit(self, reference_data_path: str) -> None:
        """
        Fits the classifier to the reference data by loading it and pre-computing embeddings.

        Args:
            reference_data_path (str): The path to the reference dataset.
        """
        print(f"Loading reference data from {reference_data_path}...")
        self.reference_data = self._load_jsonl(reference_data_path)
        
        reference_queries = [item['query'] for item in self.reference_data]
        
        print("Generating embeddings for the reference data. This may take a while...")
        self.reference_embeddings = self.model.encode(
            reference_queries,
            normalize_embeddings=True
        )
        print("Reference embeddings generated successfully.")

    def _find_top_k_neighbors(self, candidate_embeddings: np.ndarray, k: int) -> np.ndarray:
        """
        Finds the indices of the top k nearest neighbors for each candidate embedding.

        Args:
            candidate_embeddings (np.ndarray): The embeddings of the candidate problems.
            k (int): The number of nearest neighbors to find.

        Returns:
            np.ndarray: A 2D array of shape (num_candidates, k) containing the indices
                        of the nearest neighbors in the reference dataset.
        """
        # Calculate cosine similarity between candidate and reference embeddings
        similarity_matrix = candidate_embeddings @ self.reference_embeddings.T
        
        # Get the indices of the top k similarities for each candidate
        # We use np.argsort and slice the last k elements in reverse order
        top_k_indices = np.argsort(similarity_matrix, axis=1)[:, -k:][:, ::-1]
        
        return top_k_indices

    def predict(self, candidate_data_path: str, k: int, easy_threshold: int) -> List[Dict[str, Any]]:
        """
        Predicts the difficulty for a new set of candidate problems.

        Args:
            candidate_data_path (str): The path to the candidate dataset.
            k (int): The number of nearest neighbors to consider.
            easy_threshold (int): The threshold for classifying a problem as 'easy'.

        Returns:
            List[Dict[str, Any]]: The candidate data with an added 'dif' key for difficulty.
        """
        if self.reference_embeddings is None or self.reference_data is None:
            raise RuntimeError("The classifier has not been fitted. Please call .fit() first.")

        print(f"Loading candidate data from {candidate_data_path}...")
        candidate_data = self._load_jsonl(candidate_data_path)
        candidate_queries = [item['query'] for item in candidate_data]

        print("Generating embeddings for the candidate data...")
        candidate_embeddings = self.model.encode(
            candidate_queries,
            normalize_embeddings=True
        )

        print(f"Finding the {k} nearest neighbors for each candidate problem...")
        top_k_indices = self._find_top_k_neighbors(candidate_embeddings, k)

        print("Classifying problem difficulty...")
        for i, neighbor_indices in enumerate(top_k_indices):
            easy_count = 0
            for index in neighbor_indices:
                difficulty = self.reference_data[index].get('dif', 'difficult') # Default to 'difficult' if key is missing
                if difficulty == 'easy':
                    easy_count += 1
            
            # Classify based on the threshold
            if easy_count > easy_threshold:
                candidate_data[i]['dif'] = 'easy'
            else:
                candidate_data[i]['dif'] = 'difficult'
                
        return candidate_data

    @staticmethod
    def save_results(data: List[Dict[str, Any]], output_path: str) -> None:
        """
        Saves the processed data with difficulty predictions to a .jsonl file.

        Args:
            data (List[Dict[str, Any]]): The data to be saved.
            output_path (str): The path to the output .jsonl file.
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        print(f"Results have been successfully saved to {output_path}")