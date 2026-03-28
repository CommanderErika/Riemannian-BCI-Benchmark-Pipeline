from pathlib import Path
import logging
import yaml

from pipe_bci_toolkit import ProcessingPipeline

if __name__ == "__main__":

    # Configuration variables
    DATA_RAW            = "./data/raw/"
    DATA_COV            = "./data/processed/cov/"
    DATA_TS             = "./data/processed/ts/"

    # Processing Data
    processer = ProcessingPipeline(raw_dir=DATA_RAW, cov_dir=DATA_COV, ts_dir=DATA_TS)
    processer.process_all()
    




    