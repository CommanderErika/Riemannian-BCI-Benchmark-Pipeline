from abc import ABC
from typing import Optional, Dict, Any, Union, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import joblib
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from ..core.base_model import BaseModel
from ..core.base_metrics import calculate_metrics

@dataclass
class SklearnModel(BaseModel):
    """
    Base wrapper for Scikit-Learn models to standardize BCI experiment workflows.

    This class handles the entire lifecycle of a Scikit-Learn model, including:
    - Automatic Label Encoding (string labels -> integers).
    - Training and validation logic.
    - Metric calculation flattened for MLflow logging.
    - Model persistence (saving/loading).

    Attributes:
        model_class (type[BaseEstimator]): The Scikit-Learn class to instantiate (e.g., SVC).
        params (dict): Dictionary of hyperparameters to pass to the model constructor.
        model (BaseEstimator): The instantiated Scikit-Learn model object.
        encoder (LabelEncoder): Encoder to handle string-to-integer label transformations.
    """
    model_class: type[BaseEstimator]
    params: dict = field(default_factory=dict)
    model: BaseEstimator = field(init=False)
    encoder: LabelEncoder = field(default_factory=LabelEncoder, init=False)
    
    def __post_init__(self):
        """Initializes the model instance with the provided parameters."""
        self.model = self.model_class(**self.params)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Trains the model on the provided data and calculates performance metrics.

        This method automatically fits a LabelEncoder on `y_train` to handle string labels.

        Args:
            x_train (np.ndarray): Training features of shape (n_samples, n_features).
            y_train (np.ndarray): Training labels (can be strings or integers).
        """
        
        # x_train = self._process_data(x_train)
        # Label Encoding
        y_train_enc = self.encoder.fit_transform(y_train)
        # Model Training
        self.model.fit(x_train, y_train_enc)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts class labels (encoded as integers) for samples in x.

        Args:
            x (np.ndarray): Input features of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted integer labels.

        Raises:
            ValueError: If the model has not been trained yet.
        """
        
        x = self._process_data(x)
        return self.model.predict(x)

    def evaluate(self, 
                 x_test: np.ndarray, 
                 y_test: np.ndarray,
                 prefix: str = ""
                 ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Evaluates the model on a test set.

        Args:
            x_test (np.ndarray): Test features.
            y_test (np.ndarray): True test labels (strings or integers).
            prefix (str, optional): Prefix for metric keys (e.g., 'test_'). Defaults to "".

        Returns:
            Tuple containing:
                - y_test_enc (np.ndarray): Encoded ground truth labels.
                - y_pred_enc (np.ndarray): Encoded predicted labels.
                - metrics (Dict[str, float]): Dictionary of performance metrics.

        Raises:
            ValueError: If unseen labels appear in `y_test` that were not in `y_train`.
        """
            
        # x_test = self._process_data(x_test)
        
        # Transform y_test using the already trained encoder.
        y_test_enc = self.encoder.transform(y_test)

        y_pred_enc = self.predict(x_test)
        metrics = self._get_metrics(y_test_enc, y_pred_enc, prefix=prefix)

        return y_test_enc, y_pred_enc, metrics

    def save(self, path: Union[str, Path], overwrite: bool = False) -> None:
        """
        Saves the model state, including the trained model and the label encoder.

        Args:
            path (Union[str, Path]): Destination path for the .joblib file.
            overwrite (bool, optional): Whether to overwrite if file exists. Defaults to False.

        Raises:
            FileExistsError: If file exists and overwrite is False.
        """
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"File {path} already exists")
            
        # Save model, encoder, and training state
        save_data = {
            'model': self.model,
            'encoder': self.encoder,
            'params': self.params
        }

        joblib.dump(save_data, path)

    def _process_data(self, x: np.ndarray) -> np.ndarray:
        """
        Preprocesses input data before feeding it to the model.
        
        Args:
            x (np.ndarray): Raw input array.
            
        Returns:
            np.ndarray: Processed array (currently just ensures numpy format).
        """
        return np.asarray(x)

# --- Specific Model Implementations ---

class SVMModel(SklearnModel):
    """
    Wrapper for Support Vector Classifier (SVC).
    """
    def __init__(self, params: dict):
        super().__init__(model_class=SVC, params=params)

class LogisticModel(SklearnModel):
    """
    Wrapper for Logistic Regression.
    """
    def __init__(self, params: dict):
        super().__init__(model_class=LogisticRegression, params=params)

class LDAModel(SklearnModel):
    """
    Wrapper for Linear Discriminant Analysis (LDA).
    
    Note:
        LDA is considered the 'Gold Standard' for Riemannian Tangent Space classification
        in BCI due to its robustness in high-dimensional spaces when shrinkage is used.
    """
    def __init__(self, params: dict):
        super().__init__(model_class=LinearDiscriminantAnalysis, params=params)

class RidgeModel(SklearnModel):
    """
    Wrapper for Ridge Classifier.
    
    Note:
        Ridge is often faster than Linear SVM for high-dimensional data while 
        providing similar performance.
    """
    def __init__(self, params: dict):
        super().__init__(model_class=RidgeClassifier, params=params)