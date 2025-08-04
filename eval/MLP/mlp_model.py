# mlp_model.py
"""
Defines the Multilayer Perceptron (MLP) architecture using PyTorch.
"""
import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    A simple Multilayer Perceptron for classification.
    Architecture: Input -> Linear -> ReLU -> Linear -> Output
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Args:
            input_dim (int): Dimension of the input features (embedding size).
            hidden_dim (int): Dimension of the hidden layer.
            output_dim (int): Dimension of the output layer (number of classes).
        """
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),  # Add dropout for regularization
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, input_dim).
        Returns:
            torch.Tensor: The output logits of shape (batch_size, output_dim).
        """
        return self.network(x)