"""
Main entry point for the exam question classification system.
"""

import os
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.preprocessing.file_handler import QuestionFileHandler
from src.preprocessing.question_preprocessor import ExamQuestionPreprocessor
from src.models.baseline_classifier import BaselineClassifier
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ExamQuestionPipeline:
    def __init__(self, 
                 data_dir: str = "data",
                 model_dir: str = "models",
                 results_dir: str = "results",
                 test_size: float = 0.2,
                 val_size: float = 0.2,
                 random_state: int = 42,
                 classifier_config: Dict = None):
        """Initialize the pipeline."""
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.results_dir = results_dir
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        # Default classifier configuration
        self.classifier_config = classifier_config or {
            'classifier_type': 'svm',
            'feature_type': 'tfidf_ngram',
            'use_preprocessing': True,
            'feature_selection': True,
            'n_features': 5000
        }
        
        # Initialize components using existing classes
        self.file_handler = QuestionFileHandler()  # Already handles processed_dir
        self.preprocessor = ExamQuestionPreprocessor()
        self.classifier = BaselineClassifier(**self.classifier_config)
        
        # Create necessary directories
        os.makedirs(os.path.join(data_dir, "raw"), exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
    
    def process_pdf_files(self, pdf_dir: str) -> List[Dict]:
        """Process all PDF files in directory using existing file_handler."""
        logger.info(f"Processing PDF files from: {pdf_dir}")
        all_questions = []
        
        for filename in os.listdir(pdf_dir):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(pdf_dir, filename)
                # Using existing read_pdf method
                questions = self.file_handler.read_pdf(pdf_path)
                # File handler already saves processed questions
                all_questions.extend(questions)
                
        logger.info(f"Extracted total {len(all_questions)} questions")
        return all_questions
    
    def split_data(self, X: List[str], y: List[str]) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
        """Split data into train, validation, and test sets."""
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_and_evaluate(self, X_train: List[str], X_val: List[str], X_test: List[str],
                          y_train: List[str], y_train_val: List[str], y_test: List[str], 
                          hyperparameter_tuning: bool = False) -> Dict:
        """Train and evaluate using enhanced classifier methods."""
        logger.info("Training classifier...")
        
        # Using enhanced train method with hyperparameter tuning
        self.classifier.train(X_train, y_train, hyperparameter_tuning=hyperparameter_tuning)
        
        # Log best parameters if hyperparameter tuning was used
        if hyperparameter_tuning and self.classifier.best_params:
            logger.info(f"Best hyperparameters: {self.classifier.best_params}")
        
        # Using enhanced evaluate method
        train_metrics = self.classifier.evaluate(X_train, y_train)
        val_metrics = self.classifier.evaluate(X_val, y_train_val)
        test_metrics = self.classifier.evaluate(X_test, y_test)
        
        # Get feature importance if available
        feature_importance = {}
        try:
            feature_importance = self.classifier.get_feature_importance(top_n=20)
        except ValueError as e:
            logger.warning(f"Could not get feature importance: {e}")
        
        # Generate confusion matrix visualization
        try:
            y_pred = self.classifier.predict(X_test)
            self.classifier.plot_confusion_matrix(y_test, y_pred)
            logger.info(f"Confusion matrix saved to confusion_matrix.png")
        except Exception as e:
            logger.warning(f"Could not generate confusion matrix: {e}")
        
        return {
            "train_metrics": train_metrics,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "feature_importance": feature_importance
        }
    
    def visualize_results(self, metrics: Dict, timestamp: str):
        """Generate visualizations from the evaluation metrics."""
        # Create results directory for this run
        run_dir = os.path.join(self.results_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Plot accuracy comparison
        plt.figure(figsize=(10, 6))
        accuracies = {
            'Train': metrics['train_metrics']['accuracy'],
            'Validation': metrics['validation_metrics']['accuracy'],
            'Test': metrics['test_metrics']['accuracy']
        }
        
        # Add other metrics
        f1_scores = {
            'Train': metrics['train_metrics']['f1_score'],
            'Validation': metrics['validation_metrics']['f1_score'],
            'Test': metrics['test_metrics']['f1_score']
        }
        
        # Create a DataFrame for easier plotting
        df = pd.DataFrame({
            'Dataset': list(accuracies.keys()) + list(accuracies.keys()),
            'Score': list(accuracies.values()) + list(f1_scores.values()),
            'Metric': ['Accuracy'] * len(accuracies) + ['F1 Score'] * len(f1_scores)
        })
        
        # Plot grouped bar chart
        sns.barplot(x='Dataset', y='Score', hue='Metric', data=df)
        plt.title('Model Performance')
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, 'performance_metrics.png'))
        plt.close()
        
        # Plot feature importance if available
        if metrics.get('feature_importance'):
            for class_name, features in metrics['feature_importance'].items():
                if features:  # Check if features list is not empty
                    # Create DataFrame for feature importance
                    df_features = pd.DataFrame(features, columns=['Feature', 'Importance'])
                    df_features = df_features.sort_values('Importance', ascending=False).head(15)
                    
                    plt.figure(figsize=(12, 8))
                    sns.barplot(x='Importance', y='Feature', data=df_features)
                    plt.title(f'Top Features for Class: {class_name}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(run_dir, f'feature_importance_{class_name}.png'))
                    plt.close()
        
        logger.info(f"Visualizations saved to {run_dir}")
    
    def save_results(self, metrics: Dict, timestamp: str):
        """Save evaluation metrics, model, and visualizations."""
        # Save metrics
        metrics_file = os.path.join(self.model_dir, f"metrics_{timestamp}.json")
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = self._make_json_serializable(metrics)
        
        with open(metrics_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        # Using enhanced save method
        model_file = os.path.join(self.model_dir, f"model_{timestamp}.joblib")
        self.classifier.save(model_file)
        
        # Generate visualizations
        self.visualize_results(metrics, timestamp)
        
        logger.info(f"Results saved to {self.model_dir} and {self.results_dir}")
    
    def _make_json_serializable(self, obj):
        """Convert a dictionary with numpy arrays to JSON serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        else:
            return obj
    
    def run_pipeline(self, pdf_dir: str, hyperparameter_tuning: bool = False) -> Dict:
        """Run the complete pipeline with enhanced features."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Process PDF files
        raw_questions = self.process_pdf_files(pdf_dir)
        
        # 2. Preprocess questions using existing batch_process_for_classification
        logger.info("Preprocessing questions...")
        X, y = self.preprocessor.batch_process_for_classification(raw_questions)
        
        # 3. Split data
        logger.info("Splitting data into train/val/test sets...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Log data split information
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Validation set: {len(X_val)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # 4. Train and evaluate with enhanced features
        logger.info("Training and evaluating model...")
        metrics = self.train_and_evaluate(
            X_train, X_val, X_test, 
            y_train, y_val, y_test,
            hyperparameter_tuning=hyperparameter_tuning
        )
        
        # 5. Save results with enhanced visualizations
        self.save_results(metrics, timestamp)
        
        # 6. Load processed files for verification
        processed_files = self.file_handler.list_processed_files()
        logger.info(f"Generated {len(processed_files)} processed files")
        
        return metrics

def main():
    """Main function with enhanced options."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the exam question classification pipeline')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--model-dir', type=str, default='models', help='Model directory')
    parser.add_argument('--results-dir', type=str, default='results', help='Results directory')
    parser.add_argument('--pdf-dir', type=str, default='data/raw', help='PDF directory')
    parser.add_argument('--classifier-type', type=str, default='svm', 
                        choices=['svm', 'naive_bayes', 'random_forest', 'ensemble'],
                        help='Classifier type')
    parser.add_argument('--feature-type', type=str, default='tfidf_ngram',
                        choices=['tfidf', 'count', 'tfidf_ngram'],
                        help='Feature extraction method')
    parser.add_argument('--use-preprocessing', action='store_true', help='Use text preprocessing')
    parser.add_argument('--feature-selection', action='store_true', help='Use feature selection')
    parser.add_argument('--n-features', type=int, default=5000, help='Number of features for feature selection')
    parser.add_argument('--hyperparameter-tuning', action='store_true', help='Perform hyperparameter tuning')
    
    args = parser.parse_args()
    
    # Configure classifier
    classifier_config = {
        'classifier_type': args.classifier_type,
        'feature_type': args.feature_type,
        'use_preprocessing': args.use_preprocessing,
        'feature_selection': args.feature_selection,
        'n_features': args.n_features
    }
    
    # Initialize and run pipeline
    pipeline = ExamQuestionPipeline(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        results_dir=args.results_dir,
        classifier_config=classifier_config
    )
    
    metrics = pipeline.run_pipeline(
        pdf_dir=args.pdf_dir,
        hyperparameter_tuning=args.hyperparameter_tuning
    )
    
    # Print summary
    print("\nResults Summary:")
    print("Training Accuracy:", metrics["train_metrics"]["accuracy"])
    print("Validation Accuracy:", metrics["validation_metrics"]["accuracy"])
    print("Test Accuracy:", metrics["test_metrics"]["accuracy"])
    print("Training F1 Score:", metrics["train_metrics"]["f1_score"])
    print("Validation F1 Score:", metrics["validation_metrics"]["f1_score"])
    print("Test F1 Score:", metrics["test_metrics"]["f1_score"])
    
    if args.hyperparameter_tuning and pipeline.classifier.best_params:
        print("\nBest Hyperparameters:")
        for param, value in pipeline.classifier.best_params.items():
            print(f"{param}: {value}")

if __name__ == "__main__":
    main()