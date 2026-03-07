import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tracking import ExperimentOrchestrator, MLflowTracker
from src.models import SVMModel, LogisticModel, LDAModel

# First setup the server
# mlflow server --host 127.0.0.1 --port 8080

# TODO:
# create hypothesis test
# roc-auc
# Unity tests

if __name__ == "__main__":

    LIB: str        = "PyRiemann"
    DATA_DIR: str   = "./data/processed/ts" # "./data/riemanndsp/ts/"
    DATASETS: str   = ['Dreyer2023'] # ['BNCI2014_001', 'BNCI2014_002', 'BNCI2014_004', 'BNCI2015_001', 'BNCI2015_004']
    # ['BNCI2014_001.h5', 'BNCI2014_002.h5', 'BNCI2014_004.h5', 'BNCI2015_001.h5', 'BNCI2015_004.h5', 'Cho2017.h5', 'Dreyer2023.h5', 'Lee2019_MI.h5', 'Liu2024.h5', 'Weibo2014.h5', 'Zhou2016.h5']
    URI: str        = "http://127.0.0.1:8080"
    N_TRIALS        = 5
    USE_PROCRUSTES  = False

    for dataset in DATASETS:

        # Experiment Name
        EXP_NAME: str   = f'[{LIB}][{dataset}]: Classify MI with Tangent Space'

        if USE_PROCRUSTES:
            EXP_NAME = EXP_NAME + " with Procruste Alignment"

        # Tracker and Orchestrator
        mlflow_tracker  = MLflowTracker(experiment_name         =EXP_NAME)
        orchestrator    = ExperimentOrchestrator(process_lib    =LIB,
                                                 data_dir       =DATA_DIR,
                                                 use_pa         =USE_PROCRUSTES,
                                                 tracker        =mlflow_tracker,
                                                 dataset        =dataset
                                                 )
        # Registry Models

        # SVM Model
        orchestrator.registry_model(model_class     = SVMModel,
                                    name            = "SVM RBF",
                                    params          = { 'kernel' : ['rbf'],
                                                        #'C'      : (0.5, 50.0),   
                                                        # Increase cache to use more RAM (default is 200MB)
                                                        'cache_size': [2000]
                                                        })
        
        orchestrator.registry_model(model_class     = SVMModel,
                                    name            = "SVM Linear",
                                    params          = { 'kernel' : ['linear'],
                                                        #'C'      : (0.5, 50.0),   
                                                        # Increase cache to use more RAM (default is 200MB)
                                                        'cache_size': [2000]
                                                        })
        
        # Logistic Regression
        orchestrator.registry_model(model_class     = LogisticModel,
                                    name            = "Logistic_Regression",
                                    params          = { 
                                                        'penalty'     : ['l2'], # Keep simple for stability
                                                        # 'C'           : (0.1, 10.0),
                                                        'solver'      : ['lbfgs']
                                                    }                
                                    )
        
        # LDA
        orchestrator.registry_model(model_class = LDAModel,
                                    name        = "LDA",
                                    params      = { 
                                       # 'solver'    : ['lsqr', 'eigen'],
                                       # 'shrinkage' : ['auto', 0.1, 0.5, 0.9] 
                                    })

        # Running Experiments
        _ = orchestrator.run_benchmark(n_trials=N_TRIALS, n_job=5)

        del mlflow_tracker
        del orchestrator
    