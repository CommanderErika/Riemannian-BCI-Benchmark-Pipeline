import os
from typing import Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging
import yaml

from moabb.datasets.base import BaseDataset

from ..core.base_type import MOABBData
from .wrappers.moabb_config import handle_datasets, get_paradigm
from .wrappers.moabb_provider import MOABBProvider
from .io import HDF5Manager

@dataclass
class DataExtractor:
    dataset_names: list[str|Any]
    paradigm_name: str
    n_subjects: Optional[int] = None
    resample: bool = False
    freqr: Optional[float] = None
    log: bool = field(default=True)
    log_config: str = field(default="configs.yaml")

    save_to_disk: bool = False
    output_dir: Optional[Path|str] = None

    # Internal Usage
    _data: dict[str, MOABBData] = field(init=False, default_factory=dict)
    _datasets_instances: list[BaseDataset] = field(init=False)
    _paradigm_instance: Any = field(init=False)

    def __post_init__(self,):
        """Setup metadata and validation without starting heavy I/O."""

        # Logger
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        if not self.log:
            self.logger.disabled = True
            
        self.logger.debug("DataExtractor initialized with %d datasets", len(self.dataset_names))

        self._datasets_instances = handle_datasets(self.dataset_names)
        self._paradigm_instance = get_paradigm(self.paradigm_name)

        self._validate()
        self.logger.debug("DataExtractor ready. Call .download() to begin extraction.")
    
    def _validate(self):
        """Validate configuration parameters"""
        self.logger.debug("Validating configuration parameters")
        
        if self.resample and self.freqr is None:
            error_msg = "freqr must be provided when resample=True"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        if not self.resample and self.freqr is not None:
            warning_msg = "freqr is provided but resample=False - this parameter will be ignored"
            self.logger.warning(warning_msg)
        
        if not self.n_subjects is None:
            if self.n_subjects < 1:
                error_msg = f"n_subjects must be at least 1, got {self.n_subjects}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
        if self.save_to_disk and self.output_dir is None:
            raise ValueError("output_dir must be provided if save_to_disk=True")
        
    def _process_dataset(self, name: BaseDataset) -> MOABBData:
        """Download one dataset"""
        return MOABBProvider(
            dataset=name,
            paradigm=self._paradigm_instance,
            subjects=self.n_subjects,
            freqr=self.freqr if self.resample else None
        ).get_data()
    
    def _save_to_hdf5(self, dataset_name: str, data: MOABBData, out_path: Path|str):
        """Internal helper to flush data to disk immediately."""
        manager = HDF5Manager()
        file_path = Path(out_path) / f"{dataset_name}"
        # Verify path output
        os.makedirs(out_path, exist_ok=True)
        manager.save(data=data, filename=str(file_path), data_type='moabb')

    def _setup_logging(self):
        """Loads logging configuration from YAML file or sets defaults."""
        os.makedirs("logs", exist_ok=True)
        config_path = Path(self.log_config)

        if os.path.exists(config_path):
            try:
                with open(config_path, 'rt') as f:
                    config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
            except Exception as e:
                # Fallback if YAML is malformed
                logging.basicConfig(level=logging.INFO)
                print(f"Error loading {config_path}: {e}. Falling back to INFO.")
        else:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

    def download(self, save_to_disk: Optional[bool] = None, output_dir: Optional[Path|str] = None):
        """
        Triggers the sequential download of all configured datasets.
        Returns the data dictionary for convenience.
        """

        do_save = save_to_disk if save_to_disk is not None else self.save_to_disk
        out_path = output_dir if output_dir is not None else self.output_dir

        self.logger.info(f"Starting extraction for {len(self._datasets_instances)} datasets.")

        for name in self._datasets_instances:
            # Tries to download each data
            try:
                # Fetch Data
                self.logger.info(f"Processing dataset: {name}")
                current_data = self._process_dataset(name)

                if do_save:
                    self._save_to_hdf5(name.__name__, current_data, out_path)
                    self._data[name.__name__] = None
                    self.logger.info(f"Dataset {name} saved to disk and cleared from RAM.")
                else:
                    self._data[name.__name__] = current_data
                    self.logger.info(f"Dataset {name} kept in RAM.")

            except Exception as e:
                self.logger.error(f"Critical error downloading {name}: {e}", exc_info=True)

    @property
    def data(self) -> dict[str, MOABBData]:
        """Read-only access to the downloaded data."""
        return self._data.copy()  