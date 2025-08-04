# train.py
"""
Main script for training the MLP difficulty classifier.

This script performs the following steps:
1. Loads the labeled dataset.
2. Splits the data into training and validation sets.
3. Initializes the feature extractor and data loaders.
4. Defines the model, loss function, and optimizer.
5. Runs the training and validation loop for a specified number of epochs.
6. Saves the best performing model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import config
from mlp_model import MLP
from data_loader import load_jsonl, ProblemDifficultyDataset
from sentence_transformers import SentenceTransformer

def train_model():
    """Orchestrates the model training process."""
    # 1. Load data and feature extractor
    print(f"Using device: {config.DEVICE}")
    feature_extractor = SentenceTransformer(config.EMBEDDING_MODEL_PATH, device=config.DEVICE)
    full_data = load_jsonl(config.LABELED_DATA_PATH)
    
    # 2. Create PyTorch Dataset
    dataset = ProblemDifficultyDataset(full_data, feature_extractor, config.DEVICE)
    
    # 3. Split data into training and validation sets
    val_size = int(config.VALIDATION_SPLIT * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
    
    print(f"Data split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    # 4. Initialize model, loss, and optimizer
    model = MLP(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        output_dim=config.OUTPUT_DIM
    ).to(config.DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    best_val_accuracy = 0.0

    # 5. Training loop
    print("\n--- Starting Training ---")
    for epoch in range(config.EPOCHS):
        model.train()
        total_train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Training]")
        for embeddings, labels in progress_bar:
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation loop
        model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for embeddings, labels in val_loader:
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_labels, all_preds)
        
        print(
            f"Epoch {epoch+1}/{config.EPOCHS} -> "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

        # 6. Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"Best model saved to {config.MODEL_SAVE_PATH} with accuracy: {best_val_accuracy:.4f}")
            
    print("--- Training Finished ---")

if __name__ == "__main__":
    train_model()