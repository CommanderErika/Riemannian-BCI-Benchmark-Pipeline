from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path

import numpy as np
from .base_metrics import calculate_metrics

class BaseModel(ABC):

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Subclasses implement the library-specific fit here."""
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            x: Input features
            return_proba: Whether to return class probabilities (for classifiers)
            
        Returns:
            Array of predictions or probabilities
        """
        pass

    @abstractmethod
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> tuple[list, dict[str, float]]:
        """
        Evaluate the model on test data.
        
        Args:
            x_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
            overwrite: Whether to overwrite existing files
        """
        pass

    def _get_metrics(self, y_true: np.ndarray|list, y_pred: np.ndarray|list, prefix: str=''):
        return calculate_metrics(y_true, y_pred, prefix=prefix)