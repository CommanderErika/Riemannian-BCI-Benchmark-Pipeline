from pathlib import Path
import mlflow
import pandas as pd

class MLFlowReporter:
    def __init__(self, tracking_uri: str):
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)

    def get_benchmark_dataframe(self) -> pd.DataFrame:
        experiments = mlflow.search_experiments()
        all_dfs = []
        for exp in experiments:
            print(f"Scanning: {exp.name}...")
            df = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            if not df.empty:
                df.insert(0, 'experiment_name', exp.name)
                all_dfs.append(self._clean_dataframe(df))
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encapsulates the column filtering and renaming logic."""
        # Identification columns
        target_cols = ['experiment_name', 'run_id', 'status', 'start_time']
        
        # Dynamic columns
        prefixes = ('metrics.', 'params.', 'tags.')
        cols = [c for c in df.columns if c.startswith(prefixes) or c in target_cols]
        
        df = df[cols].copy()
        # Rename: 'metrics.accuracy' -> 'accuracy'
        new_columns = []
        for c in df.columns:
            if c.startswith('metrics.'):
                new_columns.append(c.replace('metrics.', ''))
            elif c.startswith('params.'):
                # We add a small 'p_' prefix to params to differentiate from metrics
                # e.g., 'params.n_splits' -> 'p_n_splits'
                new_columns.append(f"p_{c.split('.')[-1]}")
            elif c.startswith('tags.'):
                new_columns.append(c.replace('tags.', ''))
            else:
                new_columns.append(c)

        df.columns = new_columns
        return df
    
    def export_to_csv(self, output_path: str):
        """Orchestrates the fetch and the disk write."""
        df = self.get_benchmark_dataframe()
        if not df.empty:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"Report generated: {output_path}")