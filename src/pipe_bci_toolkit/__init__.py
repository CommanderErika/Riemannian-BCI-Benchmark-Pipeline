# src/pipe_bci_toolkit/__init__.py

"""
EEG Processing Pipeline - A complete toolkit for BCI research, 
integrating MOABB datasets, Riemannian geometry, and MLflow tracking.
"""

__version__ = '0.1.0'

# 1. CORE TYPES & INTERFACES
from .core import (
    MOABBData,
    CovarianceData,
    TangentSpaceData,
    BaseModel,
    DataExporter
)

# 2. DATA INGESTION & STORAGE
from .data import (
    DataExtractor,
    HDF5Manager
)

# 3. PROCESSING & PIPELINES
from .processing import ProcessingPipeline

# 4. MODELS (Batteries Included)
from .models import (
    SklearnModel,
    LogisticModel,
    SVMModel,
    LDAModel,
    RidgeModel
)

# 5. EXPERIMENT TRACKING & ORCHESTRATION
from .tracking import (
    MLflowTracker,
    ExperimentOrchestrator,
    MLFlowReporter
)


__all__ = [
    # Metadata
    '__version__',
    
    # Core
    'BaseModel',
    'MOABBData',
    'CovarianceData',
    'TangentSpaceData',
    'DataExporter',
    
    # Data
    'DataExtractor',
    'HDF5Manager',
    'MOABBProvider',
    
    # Processing
    'ProcessingPipeline',
    
    # Models
    'SklearnModel',
    'LogisticModel',
    'SVMModel',
    'LDAModel',
    'RidgeModel',
    
    # Tracking
    'MLflowTracker',
    'ExperimentOrchestrator',
    'MLFlowReporter',
    
]