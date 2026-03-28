from pathlib import Path
import logging
import yaml

from pipe_bci_toolkit import ExperimentOrchestrator, MLflowTracker
from pipe_bci_toolkit import SVMModel, LogisticModel, LDAModel

# First setup the server
# mlflow server --host 127.0.0.1 --port 8080

if __name__ == "__main__":

    LIB: str        = "PyRiemann"
    DATA_DIR: str   = "./data/processed/ts"
    DATASET: str   = 'BNCI2015_004'
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

    # SVM Model
    orchestrator.registry_model(
        model_class     = SVMModel,
        name            = "SVM RBF",
        params          = { 'kernel' : ['rbf'],
                            'cache_size': [2000]
                            }
    )
    
    orchestrator.registry_model(
        model_class     = SVMModel,
        name            = "SVM Linear",
        params          = { 
            'kernel' : ['linear'],
            'cache_size': [2000]
            }
    )
    
    # Logistic Regression
    orchestrator.registry_model(
        model_class     = LogisticModel,
        name            = "Logistic_Regression",
        params          = { 
                            'penalty'     : ['l2'],
                            'solver'      : ['lbfgs']
                        }                
    )
    
    # LDA
    orchestrator.registry_model(
        model_class = LDAModel,
        name        = "LDA",
        params      = {}
    )

    # Running Experiments
    _ = orchestrator.run_benchmark(n_trials=N_TRIALS, n_job=2)
    