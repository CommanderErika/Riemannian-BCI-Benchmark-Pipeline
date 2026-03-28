from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Union
from pathlib import Path
import logging
import yaml
import joblib

import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from pipe_bci_toolkit import ExperimentOrchestrator, MLflowTracker, BaseModel

# First setup the server
# mlflow server --host 127.0.0.1 --port 8080

@dataclass
class XGBoostModel(BaseModel):
    params: dict = field(default_factory=dict)
    encoder: LabelEncoder = field(default_factory=LabelEncoder, init=False)
    
    def __post_init__(self):
        super().__init__()
        base_params = {'verbosity': 0, 'eval_metric': 'logloss'}
        actual_params = {**base_params, **self.params}
        self.model = XGBClassifier(**actual_params)

    def fit(self, 
            x_train: np.ndarray,
            y_train: np.ndarray,
        ) -> None:
        # Label Encoding
        y_train_enc = self.encoder.fit_transform(y_train)
        #  # Model Training
        self.model.fit(x_train, y_train_enc)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)

    def evaluate(self, 
                 x_test: np.ndarray, 
                 y_test: np.ndarray,
                 prefix: str = "") -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        # Label Encoder
        y_test_enc = self.encoder.transform(y_test)
        # Predict
        y_pred_enc = self.model.predict(x_test)
        # Metrics
        metrics = self._get_metrics(y_test_enc, y_pred_enc, prefix=prefix)
        return y_test_enc, y_pred_enc, metrics

    def save(self, path: str | Path, overwrite: bool = False) -> None:
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"The file {path} already exists.")
            
        save_data = {
            'model': self.model,
            'encoder': self.encoder,
            'params': self.params
        }

        joblib.dump(save_data, path)


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
        model_class     = XGBoostModel,
        name            = "XGboost",
        params          = {}
    )

    # Running Experiments
    _ = orchestrator.run_benchmark(n_trials=N_TRIALS, n_job=2)