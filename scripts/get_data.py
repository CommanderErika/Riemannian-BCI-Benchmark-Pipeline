import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import logging

from src.data import DataExtractor
from src.data import HDF5Manager

from src.utils import handle_datasets, get_paradigm

# Configure logging once at application start
with open('configs.yaml', 'r') as f:
    config = yaml.safe_load(f)
    logging.config.dictConfig(config)

if __name__ == "__main__":

    # Configuration variables
    DATASETS            = ["BNCI2015_004"]
    PARADIGM            = "MotorImagery"
    FREQR               = 500               # Must be equal for all
    N_SUBJ              = None              # If set None, will get all data
    FORCE_SAVE          = False
    DATA_RAW            = "./data/raw/"

    # Setting Logger
    logger = logging.getLogger(__name__)
    
    for dataset in DATASETS:

        # Manager
        save_manager = HDF5Manager()
        # Getting data
        # TODO: Put the handler inside DataExtractor
        datasets = handle_datasets([dataset])
        paradigm = get_paradigm(PARADIGM)
        # TODO: Concurrent Download
        downloader = DataExtractor(dataset_names=datasets, 
                                paradigm=paradigm, 
                                n_subjects=N_SUBJ,
                                freqr=FREQR, 
                                resample=True)
        data = downloader.data
        # Saving each dataset
        #for dt in data.keys():
        print(data.keys())
        filename: str = f'{DATA_RAW}/{dataset}'
        save_manager.save(data=data[dataset], filename=filename, data_type='moabb')

        del save_manager
        del downloader