"""
Implements traditional text classification methods (TF-IDF + SVM/Naive Bayes) as baseline.
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

class BaselineClassifier:
    def __init__(self, classifier_type: str = "svm"):
        """Initialize the baseline classifier."""
        # Set up feature extractor
        self.feature_extractor = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            min_df=2,
            max_df=0.85
        )
        
        # Set up classifier
        if classifier_type == "svm":
            self.classifier = SVC(
                kernel='linear', 
                probability=True,
                class_weight='balanced'
            )
        elif classifier_type == "naive_bayes":
            self.classifier = MultinomialNB(alpha=0.1)
        else:
            raise ValueError("Classifier type must be 'svm' or 'naive_bayes'")
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('features', self.feature_extractor),
            ('classifier', self.classifier)
        ])
        
        self.is_fitted = False
    
    def train(self, X_train: List[str], y_train: List[str]):
        """Train the classifier."""
        self.pipeline.fit(X_train, y_train)
        self.is_fitted = True
    
    def predict(self, X_test: List[str]) -> List[str]:
        """Predict labels for test data."""
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        return self.pipeline.predict(X_test)
    
    def predict_proba(self, X_test: List[str]) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        return self.pipeline.predict_proba(X_test)
    
    def evaluate(self, X_test: List[str], y_test: List[str]) -> Dict:
        """Evaluate the classifier."""
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        y_pred = self.predict(X_test)
        
        return {
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "accuracy": accuracy_score(y_test, y_pred)
        }
    
    def save(self, file_path: str):
        """Save the model."""
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        joblib.dump(self.pipeline, file_path)
    
    @classmethod
    def load(cls, file_path: str) -> 'BaselineClassifier':
        """Load a trained model."""
        instance = cls()
        instance.pipeline = joblib.load(file_path)
        instance.is_fitted = True
        return instance
    

