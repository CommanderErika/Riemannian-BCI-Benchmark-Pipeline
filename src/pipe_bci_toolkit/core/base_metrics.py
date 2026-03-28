from typing import Dict
import numpy as np
from sklearn.metrics import classification_report

def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = ''
    ) -> Dict[str, float]:
    """
    Calculates classification metrics and flattens the structure for MLflow.

    It computes Accuracy, Macro F1/Precision/Recall, and Weighted F1/Precision/Recall.

    Args:
        y_true (np.ndarray): Encoded true labels.
        y_pred (np.ndarray): Encoded predicted labels.
        prefix (str, optional): Prefix to add to metric keys.

    Returns:
        Dict[str, float]: Flattened dictionary, e.g., {'test_f1_macro': 0.85}.
    """
    # Generate classification report as a dictionary
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    flat_metrics = {}
    
    # Extract global accuracy
    flat_metrics[f'{prefix}accuracy'] = report['accuracy']
    
    # Extract averages (macro and weighted are standard for multi-class BCI)
    for avg_type in ['macro avg', 'weighted avg']:
        clean_name = avg_type.replace(' avg', '') # becomes 'macro' or 'weighted'
        for metric, value in report[avg_type].items():
            if metric != 'support': # Support is just the count, not a performance metric
                flat_metrics[f'{prefix}{metric}_{clean_name}'] = value            
    return flat_metrics