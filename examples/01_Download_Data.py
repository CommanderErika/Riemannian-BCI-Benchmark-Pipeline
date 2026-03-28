from pipe_bci_toolkit import DataExtractor

if __name__ == "__main__":

    # Configuration variables
    DATASET             = ["BNCI2015_004"]
    PARADIGM            = "MotorImagery"
    FREQR               = 500               # If freq is set, then it is resampled.
    N_SUBJ              = 2                 # If set None, will get data for all subjects.
    FORCE_SAVE          = False
    DATA_RAW            = "./data/raw/"

    # Getting data
    downloader = DataExtractor(
                    dataset_names=DATASET, 
                    paradigm_name=PARADIGM,
                    n_subjects=N_SUBJ,
                    freqr=FREQR, 
                    resample=True
                )
    
    data = downloader.download(save_to_disk=True, output_dir="data/raw")