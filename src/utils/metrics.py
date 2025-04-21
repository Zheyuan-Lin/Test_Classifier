"""
Utility functions for calculating various evaluation metrics.
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def calculate_reliability_diagram(confidences: List[float], correct: List[bool], 
                                num_bins: int = 10) -> Dict:
    """Calculate reliability diagram data for calibration analysis."""
    bin_confidences = [[] for _ in range(num_bins)]
    bin_accuracies = [[] for _ in range(num_bins)]
    
    for conf, corr in zip(confidences, correct):
        bin_idx = min(int(conf * num_bins), num_bins - 1)
        bin_confidences[bin_idx].append(conf)
        bin_accuracies[bin_idx].append(float(corr))
    
    avg_confidences = []
    avg_accuracies = []
    
    for bin_conf, bin_acc in zip(bin_confidences, bin_accuracies):
        if bin_conf:
            avg_confidences.append(sum(bin_conf) / len(bin_conf))
            avg_accuracies.append(sum(bin_acc) / len(bin_acc))
    
    return {
        "confidences": avg_confidences,
        "accuracies": avg_accuracies
    }

def calculate_expected_calibration_error(confidences: List[float], correct: List[bool], 
                                      num_bins: int = 10) -> float:
    """Calculate the expected calibration error."""
    bin_confidences = [[] for _ in range(num_bins)]
    bin_accuracies = [[] for _ in range(num_bins)]
    
    for conf, corr in zip(confidences, correct):
        bin_idx = min(int(conf * num_bins), num_bins - 1)
        bin_confidences[bin_idx].append(conf)
        bin_accuracies[bin_idx].append(float(corr))
    
    ece = 0.0
    total_samples = len(confidences)
    
    for bin_conf, bin_acc in zip(bin_confidences, bin_accuracies):
        if bin_conf:
            bin_size = len(bin_conf)
            bin_conf_avg = sum(bin_conf) / bin_size
            bin_acc_avg = sum(bin_acc) / bin_size
            ece += (bin_size / total_samples) * abs(bin_conf_avg - bin_acc_avg)
    
    return ece

def find_confused_category_pairs(conf_matrix: np.ndarray, categories: List[str], 
                               top_k: int = 5) -> List[Tuple[str, str, int]]:
    """Find the most confused category pairs from confusion matrix."""
    category_pairs = []
    
    for i in range(len(categories)):
        for j in range(i+1, len(categories)):
            cat1, cat2 = categories[i], categories[j]
            confusion = conf_matrix[i, j] + conf_matrix[j, i]
            if confusion > 0:
                category_pairs.append((cat1, cat2, confusion))
    
    return sorted(category_pairs, key=lambda x: x[2], reverse=True)[:top_k]

def calculate_per_category_metrics(y_true: List[str], y_pred: List[str], 
                                 categories: List[str]) -> Dict:
    """Calculate detailed metrics for each category."""
    report = classification_report(y_true, y_pred, output_dict=True)
    
    return {
        cat: {
            "precision": report.get(cat, {}).get("precision", 0),
            "recall": report.get(cat, {}).get("recall", 0),
            "f1-score": report.get(cat, {}).get("f1-score", 0),
            "support": report.get(cat, {}).get("support", 0)
        } for cat in categories
    } 