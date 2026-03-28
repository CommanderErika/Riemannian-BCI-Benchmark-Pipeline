from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Union
import joblib
from pathlib import Path
import logging
import yaml

import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
from pipe_bci_toolkit import ExperimentOrchestrator, MLflowTracker, BaseModel

# First setup the server
# mlflow server --host 127.0.0.1 --port 8080

@dataclass
class TFBCIModel(BaseModel):
    params: dict = field(default_factory=dict)
    encoder: LabelEncoder = field(default_factory=LabelEncoder, init=False)

    def _create_model(self) -> tf.keras.Model:
        """
        Defines the Neural Network architecture. 
        You can override this method to create different BCI nets (EEGNet, CNN, etc.)
        """
        model = models.Sequential([
            # If the input is 2D (Channels, Time), we flatten it first
            layers.Input(shape=self.input_shape),
            layers.Flatten(),
            
            # Hidden Layer 1
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            
            # Hidden Layer 2
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            
            # Hidden Layer 3
            layers.Dense(32, activation='relu'),
            
            # Output Head
            layers.Dense(self.n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, epochs=10, batch_size=16, validation_data=None):
        # Settings
        self.input_shape = x_train.shape[1:]
        self.n_classes = len(np.unique(y_train))
        print(f"Number of classes: {self.n_classes}")
        # Create Model
        self.model = self._create_model()
        
        # Label Encoding
        y_train_enc = self.encoder.fit_transform(y_train)
        
        # Validation Logic
        val_payload = None
        if validation_data:
            x_v, y_v = validation_data
            y_v_enc = self.encoder.transform(y_v)
            val_payload = (x_v, y_v_enc)

        # TF Fit
        history = self.model.fit(
            x_train, y_train_enc,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_payload,
            verbose=1
        )
        
        # return history.history

    def predict(self, x: np.ndarray) -> np.ndarray:
        preds_proba = self.model.predict(x, verbose=0)
        return np.argmax(preds_proba, axis=1)

    def evaluate(self, 
                 x_test: np.ndarray, 
                 y_test: np.ndarray,
                 prefix: str = "") -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        
        y_test_enc = self.encoder.transform(y_test)
        y_pred_enc = self.predict(x_test)
        # Metrics
        metrics = self._get_metrics(y_test_enc, y_pred_enc, prefix=prefix)
        return y_test_enc, y_pred_enc, metrics

    def save(self, path: str|Path, overwrite: bool = False):
        path = Path(path)
        # Save TF Model (Weights + Arch)
        self.model.save(path.with_suffix('.h5'))
        # Save Metadata (Encoder + State)
        metadata = {
            'encoder': self.encoder,
            'is_trained': self._is_trained,
            'input_shape': self.input_shape,
            'n_classes': self.n_classes,
            'params': self.params
        }
        joblib.dump(metadata, path.with_suffix('.pkt'))

if __name__ == "__main__":

    LIB: str        = "PyRiemann"
    DATA_DIR: str   = "./data/processed/ts"
    DATASET: str    = 'BNCI2015_004'
    URI: str        = "http://127.0.0.1:8080"
    N_TRIALS        = 1
    USE_PROCRUSTES  = False

    # Experiment Name
    EXP_NAME: str   = f'[{LIB}][{DATASET}]: Classify MI with Tangent Space'

    if USE_PROCRUSTES:
        EXP_NAME = EXP_NAME + " with Procruste Alignment"

    # Tracker and Orchestrator
    mlflow_tracker  = MLflowTracker(experiment_name=EXP_NAME)
    orchestrator    = ExperimentOrchestrator(
        process_lib    =LIB,
        data_dir       =DATA_DIR,
        use_pa         =USE_PROCRUSTES,
        tracker        =mlflow_tracker,
        dataset        =DATASET
    )

    orchestrator.registry_model(
        model_class     = TFBCIModel,
        name            = "TF simple model",
        params          = {}
    )

    # Running Experiments
    _ = orchestrator.run_benchmark(n_trials=N_TRIALS, n_job=2)