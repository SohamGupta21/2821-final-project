"""Simple feedforward neural network classifier."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split


class SimpleNNClassifier(nn.Module):
    """Simple feedforward neural network for binary classification."""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [64, 32], num_classes: int = 2):
        """
        Initialize the neural network.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of output classes
        """
        super(SimpleNNClassifier, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)
    
    def predict(self, x: np.ndarray) -> int:
        """
        Predict class for input.
        
        Args:
            x: Input feature vector (1D or 2D array)
        
        Returns:
            Predicted class index
        """
        self.eval()
        with torch.no_grad():
            if len(x.shape) == 1:
                x = x.reshape(1, -1)
            x_tensor = torch.FloatTensor(x)
            output = self.forward(x_tensor)
            _, predicted = torch.max(output, 1)
            return predicted.item()
    
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            x: Input feature vector (1D or 2D array)
        
        Returns:
            Class probabilities
        """
        self.eval()
        with torch.no_grad():
            if len(x.shape) == 1:
                x = x.reshape(1, -1)
            x_tensor = torch.FloatTensor(x)
            output = self.forward(x_tensor)
            proba = torch.softmax(output, dim=1)
            return proba.numpy()
    
    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training labels
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            validation_split: Fraction of data to use for validation
            verbose: Whether to print training progress
        
        Returns:
            Dictionary with training history
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        
        # Setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        
        # Training loop
        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            
            # Mini-batch training
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.eval()
            with torch.no_grad():
                val_outputs = self.forward(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
                _, val_predicted = torch.max(val_outputs, 1)
                val_acc = (val_predicted == y_val_tensor).float().mean().item()
            
            history["train_loss"].append(train_loss / (len(X_train_tensor) // batch_size + 1))
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {history['train_loss'][-1]:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}")
        
        return history


def generate_synthetic_loan_data(
    n_samples: int = 1000,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate synthetic loan approval data.
    
    Returns:
        X: Feature matrix
        y: Labels (0=denied, 1=approved)
        feature_names: List of feature names
    """
    np.random.seed(random_state)
    
    n_samples = n_samples
    feature_names = ["income", "credit_score", "age", "employment_years"]
    
    # Generate features
    income = np.random.normal(50000, 20000, n_samples)
    credit_score = np.random.normal(650, 100, n_samples)
    age = np.random.normal(35, 10, n_samples)
    employment_years = np.random.exponential(5, n_samples)
    
    # Normalize features
    income = (income - income.mean()) / income.std()
    credit_score = (credit_score - credit_score.mean()) / credit_score.std()
    age = (age - age.mean()) / age.std()
    employment_years = (employment_years - employment_years.mean()) / employment_years.std()
    
    X = np.column_stack([income, credit_score, age, employment_years])
    
    # Generate labels based on simple rules (for ground truth)
    # Approved if: high income AND high credit_score
    y = ((income > 0.5) & (credit_score > 0.3)).astype(int)
    
    # Add some noise
    noise = np.random.random(n_samples) < 0.1
    y[noise] = 1 - y[noise]
    
    return X, y, feature_names


