"""PyTorch-based neural network models for tabular data.

This module provides PyTorch implementations with sklearn-compatible API.
"""
from __future__ import annotations

import numpy as np


class TorchTabularClassifier:
    """PyTorch binary classifier for tabular data with sklearn-like API.
    
    Features:
    - Multi-layer feedforward network
    - Dropout regularization
    - Optional focal loss for imbalanced data
    - Early stopping on training loss
    """
    
    def __init__(
        self,
        in_dim: int | None = None,
        hidden: int = 64,
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 64,
        focal_loss: bool = False,
    ):
        """Initialize neural network classifier.
        
        Args:
            in_dim: Input dimension (auto-detected from data if None)
            hidden: Hidden layer size
            dropout: Dropout rate
            lr: Learning rate
            epochs: Number of training epochs
            batch_size: Batch size for training
            focal_loss: Whether to use focal loss (for imbalanced data)
        """
        self.in_dim = in_dim
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.focal = focal_loss
        self.model = None
        self.device = None
        self.is_fitted_ = False
    
    def _criterion(self, logits, y):
        """Compute loss (BCE or focal loss)."""
        import torch
        import torch.nn as nn
        
        if not self.focal:
            return nn.BCEWithLogitsLoss()(logits, y)
        
        # Focal loss for imbalanced data
        p = torch.sigmoid(logits)
        pt = p * y + (1 - p) * (1 - y)
        gamma = 2.0
        alpha = 0.25
        return (-(alpha * (1 - pt) ** gamma * torch.log(pt + 1e-8))).mean()
    
    def fit(self, X, y):
        """Fit model to training data.
        
        Args:
            X: Training features (array-like)
            y: Training labels (array-like)
            
        Returns:
            self
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        
        # Auto-detect input dimension
        if self.in_dim is None:
            self.in_dim = X.shape[1]
        
        # Build model
        if self.model is None:
            self.model = nn.Sequential(
                nn.Linear(self.in_dim, self.hidden),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden, self.hidden),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden, 1),
            )
        
        # Prepare data
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(
            y.values if hasattr(y, "values") else y,
            dtype=torch.float32
        ).view(-1, 1)
        
        dataset = TensorDataset(X_t, y_t)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Setup training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Training loop with early stopping
        best_loss = float("inf")
        best_state = None
        
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            
            for xb, yb in dataloader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = self._criterion(logits, yb)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * xb.size(0)
            
            epoch_loss = running_loss / len(dataset)
            
            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = self.model.state_dict()
        
        # Load best weights
        if best_state:
            self.model.load_state_dict(best_state)
        
        self.is_fitted_ = True
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities.
        
        Args:
            X: Features (array-like)
            
        Returns:
            Array of shape (n_samples, 2) with probabilities for each class
        """
        import torch
        
        self.model.eval()
        
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(
                self.device if self.device is not None else 'cpu'
            )
            logits = self.model(X_t)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        
        return np.vstack([1 - probs, probs]).T
    
    def predict(self, X):
        """Predict class labels.
        
        Args:
            X: Features (array-like)
            
        Returns:
            Array of predicted class labels (0 or 1)
        """
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
    
    def get_params(self, deep: bool = True):
        """Get model parameters (sklearn compatibility).
        
        Args:
            deep: Whether to return deep copy
            
        Returns:
            Dictionary of parameters
        """
        return {
            "in_dim": self.in_dim,
            "hidden": self.hidden,
            "dropout": self.dropout,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "focal_loss": self.focal,
        }
    
    def set_params(self, **params):
        """Set model parameters (sklearn compatibility).
        
        Args:
            **params: Parameters to set
            
        Returns:
            self
        """
        for k, v in params.items():
            setattr(self, k, v)
        
        # Reset model to rebuild with new parameters
        self.model = None
        
        return self
