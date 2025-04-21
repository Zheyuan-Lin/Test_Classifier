"""
Visualization utilities for the exam classification pipeline.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

class ResultVisualizer:
    """Class for visualizing classification results."""
    
    def __init__(self, output_dir: str = "./figures"):
        """Initialize the visualizer with output directory."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def plot_confusion_matrix(self, y_true: List[str], y_pred: List[str], 
                            categories: List[str], title: str = "Confusion Matrix",
                            filename: str = "confusion_matrix.png"):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=categories,
                   yticklabels=categories)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path)
        plt.close()
    
    def plot_roc_curves(self, y_true: List[str], y_pred_proba: np.ndarray,
                       categories: List[str], filename: str = "roc_curves.png"):
        """Plot ROC curves for each category."""
        # Binarize the labels
        y_true_bin = label_binarize(y_true, classes=categories)
        
        plt.figure(figsize=(10, 8))
        
        for i, category in enumerate(categories):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{category} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Each Category')
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path)
        plt.close()
    
    def plot_reliability_diagram(self, confidences: List[float], correct: List[bool],
                               num_bins: int = 10, filename: str = "reliability_diagram.png"):
        """Plot reliability diagram for calibration analysis."""
        bin_confidences = [[] for _ in range(num_bins)]
        bin_accuracies = [[] for _ in range(num_bins)]
        
        for conf, corr in zip(confidences, correct):
            bin_idx = min(int(conf * num_bins), num_bins - 1)
            bin_confidences[bin_idx].append(conf)
            bin_accuracies[bin_idx].append(float(corr))
        
        avg_confidences = []
        avg_accuracies = []
        bin_sizes = []
        
        for bin_conf, bin_acc in zip(bin_confidences, bin_accuracies):
            if bin_conf:
                avg_confidences.append(sum(bin_conf) / len(bin_conf))
                avg_accuracies.append(sum(bin_acc) / len(bin_acc))
                bin_sizes.append(len(bin_conf))
        
        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.plot(avg_confidences, avg_accuracies, 'o-', label='Model Calibration')
        
        # Add bin sizes as point sizes
        plt.scatter(avg_confidences, avg_accuracies, s=bin_sizes, alpha=0.5)
        
        plt.xlabel('Predicted Confidence')
        plt.ylabel('Actual Accuracy')
        plt.title('Reliability Diagram')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path)
        plt.close()
    
    def plot_category_distribution(self, y_true: List[str], y_pred: List[str],
                                 categories: List[str], filename: str = "category_distribution.png"):
        """Plot distribution of true and predicted categories."""
        true_counts = [y_true.count(cat) for cat in categories]
        pred_counts = [y_pred.count(cat) for cat in categories]
        
        x = np.arange(len(categories))
        width = 0.35
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, true_counts, width, label='True', alpha=0.7)
        plt.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.7)
        
        plt.xlabel('Categories')
        plt.ylabel('Count')
        plt.title('Distribution of True vs Predicted Categories')
        plt.xticks(x, categories, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path)
        plt.close()
    
    def plot_performance_metrics(self, metrics: Dict[str, float],
                               filename: str = "performance_metrics.png"):
        """Plot performance metrics as a bar chart."""
        plt.figure(figsize=(10, 6))
        plt.bar(metrics.keys(), metrics.values())
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Model Performance Metrics')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path)
        plt.close()
    
    def plot_training_history(self, history: Dict[str, List[float]],
                            filename: str = "training_history.png"):
        """Plot training history (loss, accuracy, etc.)."""
        plt.figure(figsize=(12, 6))
        
        for metric, values in history.items():
            plt.plot(values, label=metric)
        
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path)
        plt.close()
    
    def create_summary_report(self, results: Dict, output_dir: str = None):
        """Create a comprehensive visualization report."""
        if output_dir is None:
            output_dir = self.output_dir
        
        # Create subplots for different visualizations
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Confusion Matrix
        plt.subplot(2, 2, 1)
        cm = confusion_matrix(results['y_true'], results['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=results['categories'],
                   yticklabels=results['categories'])
        plt.title('Confusion Matrix')
        
        # Plot 2: ROC Curves
        plt.subplot(2, 2, 2)
        y_true_bin = label_binarize(results['y_true'], classes=results['categories'])
        for i, category in enumerate(results['categories']):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], results['y_pred_proba'][:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{category} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title('ROC Curves')
        plt.legend(loc='lower right')
        
        # Plot 3: Reliability Diagram
        plt.subplot(2, 2, 3)
        self.plot_reliability_diagram(results['confidences'], results['correct'])
        
        # Plot 4: Category Distribution
        plt.subplot(2, 2, 4)
        true_counts = [results['y_true'].count(cat) for cat in results['categories']]
        pred_counts = [results['y_pred'].count(cat) for cat in results['categories']]
        x = np.arange(len(results['categories']))
        width = 0.35
        plt.bar(x - width/2, true_counts, width, label='True', alpha=0.7)
        plt.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.7)
        plt.title('Category Distribution')
        plt.xticks(x, results['categories'], rotation=45, ha='right')
        plt.legend()
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, "summary_report.png")
        plt.savefig(output_path)
        plt.close() 