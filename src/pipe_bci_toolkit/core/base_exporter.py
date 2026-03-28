from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd

class DataExporter(ABC):
    @abstractmethod
    def save(self, df: pd.DataFrame, destination: str):
        pass