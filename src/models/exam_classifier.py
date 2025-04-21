"""
Main classifier class that combines multiple classification approaches
and implements the full exam-specific framework.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from datetime import datetime
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score

from src.config.constants import CURRICULUM_CATEGORIES, DEFAULT_CONFIDENCE_THRESHOLD
from src.preprocessing.question_preprocessor import ExamQuestionPreprocessor
from src.models.baseline_classifier import BaselineClassifier
from src.models.transformer_classifier import TransformerClassifier
from src.models.gpt4_classifier import GPT4ZeroShotClassifier

class ExamQuestionClassifier:
    def __init__(self, 
                 baseline_model: BaselineClassifier = None, 
                 transformer_model: TransformerClassifier = None,
                 gpt4_model: GPT4ZeroShotClassifier = None,
                 ensemble_method: str = "weighted_vote",
                 model_weights: Dict[str, float] = None,
                 confidence_thresholds: Dict[str, float] = None,
                 save_dir: str = "models/exam_classifier"):
        """
        Initialize with optional pre-trained models.
        
        Args:
            baseline_model: Pre-trained baseline model
            transformer_model: Pre-trained transformer model
            gpt4_model: Pre-trained GPT-4 model
            ensemble_method: Method for combining model predictions 
                             ("weighted_vote", "max_confidence", "cascade")
            model_weights: Dictionary of model weights for weighted voting
                           e.g., {"transformer": 0.7, "gpt4": 0.2, "baseline": 0.1}
            confidence_thresholds: Dictionary of confidence thresholds for each model
                                  e.g., {"transformer": 0.8, "gpt4": 0.7, "baseline": 0.6}
            save_dir: Directory to save models and results
        """
        self.baseline_model = baseline_model
        self.transformer_model = transformer_model
        self.gpt4_model = gpt4_model
        self.preprocessor = ExamQuestionPreprocessor()
        
        # Ensemble configuration
        self.ensemble_method = ensemble_method
        
        # Default model weights if not provided
        if model_weights is None:
            self.model_weights = {
                "transformer": 0.6,
                "gpt4_zero_shot": 0.3,
                "baseline": 0.1
            }
        else:
            self.model_weights = model_weights
        
        # Default confidence thresholds if not provided
        if confidence_thresholds is None:
            self.confidence_thresholds = {
                "transformer": 0.8,
                "gpt4_zero_shot": 0.7,
                "baseline": 0.6
            }
        else:
            self.confidence_thresholds = confidence_thresholds
        
        # Create save directory if it doesn't exist
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Performance tracking
        self.performance_history = {
            "baseline": [],
            "transformer": [],
            "gpt4_zero_shot": [],
            "ensemble": []
        }
        
    def train_baseline(self, questions_df: pd.DataFrame, test_size: float = 0.2, 
                      random_state: int = 42, classifier_type: str = "svm", 
                      hyperparameter_tuning: bool = True):
        """
        Train the baseline model with enhanced options.
        
        Args:
            questions_df: DataFrame of questions
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            classifier_type: Type of classifier to use ("svm", "nb", "rf", "lr")
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Evaluation results and data splits
        """
        # Prepare data
        X = questions_df["processed_stem"].tolist()
        y = questions_df["curriculum_category"].tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Initialize and train baseline model if not provided
        if not self.baseline_model:
            self.baseline_model = BaselineClassifier(classifier_type=classifier_type)
        
        # Train with hyperparameter tuning if requested
        self.baseline_model.train(
            X_train, 
            y_train, 
            use_hyperparameter_tuning=hyperparameter_tuning
        )
        
        # Evaluate
        eval_results = self.baseline_model.evaluate(X_test, y_test)
        
        # Save performance history
        self.performance_history["baseline"].append({
            "timestamp": datetime.now().isoformat(),
            "metrics": eval_results,
            "model_config": {
                "classifier_type": classifier_type,
                "hyperparameter_tuning": hyperparameter_tuning
            }
        })
        
        return eval_results, (X_train, X_test, y_train, y_test)
    
    def train_transformer(self, questions_df: pd.DataFrame, test_size: float = 0.2, 
                         random_state: int = 42, model_name: str = "allenai/scibert_scivocab_uncased",
                         num_epochs: int = 3, batch_size: int = 8, learning_rate: float = 2e-5,
                         use_data_augmentation: bool = True):
        """
        Train the transformer model with enhanced options.
        
        Args:
            questions_df: DataFrame of questions
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            model_name: Name of pre-trained transformer model
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for training
            use_data_augmentation: Whether to use data augmentation
            
        Returns:
            Evaluation results and data splits
        """
        # Prepare data
        X = questions_df["processed_stem"].tolist()
        y = questions_df["curriculum_category"].tolist()
        
        # Get explanations if available
        explanations = None
        if "explanation" in questions_df.columns:
            explanations = questions_df["explanation"].tolist()
        
        # Split data
        if explanations:
            X_train, X_test, y_train, y_test, exp_train, exp_test = train_test_split(
                X, y, explanations, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            exp_train, exp_test = None, None
        
        # Initialize transformer model if not provided
        if not self.transformer_model:
            self.transformer_model = TransformerClassifier(
                model_name=model_name,
                num_labels=len(CURRICULUM_CATEGORIES)
            )
        
        # Train the model with enhanced options
        self.transformer_model.train(
            train_texts=X_train,
            train_labels=y_train,
            eval_texts=X_test,
            eval_labels=y_test,
            train_explanations=exp_train,
            eval_explanations=exp_test,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            use_data_augmentation=use_data_augmentation
        )
        
        # Evaluate
        eval_results = self.transformer_model.evaluate(X_test, y_test, explanations=exp_test)
        
        # Save performance history
        self.performance_history["transformer"].append({
            "timestamp": datetime.now().isoformat(),
            "metrics": eval_results,
            "model_config": {
                "model_name": model_name,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "use_data_augmentation": use_data_augmentation
            }
        })
        
        return eval_results, (X_train, X_test, y_train, y_test)
    
    def classify(self, question_text: str, use_explanation: bool = True, 
                confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
                use_ensemble: bool = True) -> Dict:
        """
        Classify a single exam question using the appropriate model based on confidence.
        
        Args:
            question_text: Text of the question to classify
            use_explanation: Whether to use explanation text if available
            confidence_threshold: Confidence threshold for model selection
            use_ensemble: Whether to use ensemble methods for prediction
            
        Returns:
            Dictionary with classification results
        """
        # Preprocess the question
        processed_data = self.preprocessor.preprocess({"question_text": question_text})
        stem = processed_data["processed_stem"]
        
        # Add explanation if available and requested
        if use_explanation and processed_data.get("explanation"):
            text_to_classify = f"{stem}\n\nExplanation: {processed_data['explanation']}"
        else:
            text_to_classify = stem
        
        result = {
            "original_question": question_text,
            "processed_stem": stem,
            "model_used": None,
            "predicted_category": None,
            "confidence": 0.0,
            "all_predictions": {}
        }
        
        # Collect predictions from all available models
        model_predictions = {}
        
        # Try transformer model if available
        if self.transformer_model:
            predicted_labels, probabilities = self.transformer_model.predict([text_to_classify])
            max_prob = np.max(probabilities[0])
            
            transformer_result = {
                "predicted_category": predicted_labels[0],
                "confidence": float(max_prob),
                "all_predictions": {
                    CURRICULUM_CATEGORIES[i]: float(prob) 
                    for i, prob in enumerate(probabilities[0])
                }
            }
            
            model_predictions["transformer"] = transformer_result
        
        # Try GPT-4 model if available
        if self.gpt4_model:
            gpt_results = self.gpt4_model.classify([text_to_classify])[0]
            
            gpt4_result = {
                "predicted_category": gpt_results["predicted_category"],
                "confidence": gpt_results["confidence"],
                "reasoning": gpt_results["reasoning"]
            }
            
            model_predictions["gpt4_zero_shot"] = gpt4_result
        
        # Try baseline model if available
        if self.baseline_model:
            predictions = self.baseline_model.predict([text_to_classify])
            probabilities = self.baseline_model.predict_proba([text_to_classify])[0]
            max_prob_idx = np.argmax(probabilities)
            
            baseline_result = {
                "predicted_category": predictions[0],
                "confidence": float(probabilities[max_prob_idx]),
                "all_predictions": {
                    label: float(probabilities[i]) 
                    for i, label in enumerate(self.baseline_model.pipeline.classes_)
                }
            }
            
            model_predictions["baseline"] = baseline_result
        
        # Use ensemble method if requested and multiple models are available
        if use_ensemble and len(model_predictions) > 1:
            ensemble_result = self._get_ensemble_prediction(model_predictions)
            
            # Update result with ensemble prediction
            result.update(ensemble_result)
            
            # Add individual model predictions for reference
            result["model_predictions"] = model_predictions
        else:
            # Fall back to cascade strategy if ensemble not used
            # Try transformer first
            if "transformer" in model_predictions:
                pred = model_predictions["transformer"]
                if pred["confidence"] >= self.confidence_thresholds.get("transformer", confidence_threshold):
                    result.update(pred)
                    result["model_used"] = "transformer"
                    return result
            
            # Try GPT-4 next
            if "gpt4_zero_shot" in model_predictions:
                pred = model_predictions["gpt4_zero_shot"]
                if pred["confidence"] >= self.confidence_thresholds.get("gpt4_zero_shot", confidence_threshold):
                    result.update(pred)
                    result["model_used"] = "gpt4_zero_shot"
                    return result
            
            # Fall back to baseline
            if "baseline" in model_predictions:
                pred = model_predictions["baseline"]
                result.update(pred)
                result["model_used"] = "baseline"
            
            # If no models available
            if not model_predictions:
                result["predicted_category"] = "Unknown"
                result["confidence"] = 0.0
                result["model_used"] = "none"
        
        return result
    
    def batch_classify(self, questions: List[str], use_explanation: bool = True,
                      confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
                      use_ensemble: bool = True, batch_size: int = 16,
                      show_progress: bool = True) -> List[Dict]:
        """
        Classify a batch of questions.
        
        Args:
            questions: List of questions to classify
            use_explanation: Whether to use explanation text if available
            confidence_threshold: Confidence threshold for model selection
            use_ensemble: Whether to use ensemble methods for prediction
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            List of dictionaries with classification results
        """
        results = []
        
        # Process in batches for efficiency
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            batch_results = [
                self.classify(q, use_explanation, confidence_threshold, use_ensemble) 
                for q in batch
            ]
            results.extend(batch_results)
            
            # Show progress if requested
            if show_progress:
                print(f"Processed {min(i + batch_size, len(questions))}/{len(questions)} questions")
        
        return results
    
    def evaluate_distractor_robustness(self, questions_df: pd.DataFrame) -> Dict:
        """
        Evaluate the model's robustness to distractors in MCQs.
        
        This compares classification accuracy when using only the stem vs.
        using the full question text including options.
        """
        # Get MCQs only
        mcqs = questions_df[questions_df["question_type"] == "MCQ"]
        
        # Classify using stems only
        stem_predictions = self.batch_classify(mcqs["processed_stem"].tolist())
        stem_predicted_categories = [pred["predicted_category"] for pred in stem_predictions]
        
        # Classify using full questions (stem + options)
        full_text = [
            f"{row['processed_stem']} {' '.join(row['processed_options'])}" 
            for _, row in mcqs.iterrows()
        ]
        full_predictions = self.batch_classify(full_text)
        full_predicted_categories = [pred["predicted_category"] for pred in full_predictions]
        
        # Compare results
        true_categories = mcqs["curriculum_category"].tolist()
        stem_accuracy = accuracy_score(true_categories, stem_predicted_categories)
        full_accuracy = accuracy_score(true_categories, full_predicted_categories)
        
        # Calculate agreement between stem-only and full-text predictions
        agreement = sum(1 for s, f in zip(stem_predicted_categories, full_predicted_categories) if s == f) / len(mcqs)
        
        return {
            "stem_only_accuracy": stem_accuracy,
            "full_text_accuracy": full_accuracy,
            "stem_full_agreement": agreement,
            "distractor_influence": 1 - agreement  # Higher means distractors influence predictions more
        }
    
    def evaluate_cross_format_accuracy(self, questions_df: pd.DataFrame) -> Dict:
        """
        Evaluate the model's performance across different question formats.
        
        This compares classification accuracy on MCQs vs. open-ended questions.
        """
        # Split by question type
        mcqs = questions_df[questions_df["question_type"] == "MCQ"]
        open_ended = questions_df[questions_df["question_type"] == "Open-ended"]
        
        # Evaluate MCQs
        mcq_predictions = self.batch_classify(mcqs["processed_stem"].tolist())
        mcq_predicted_categories = [pred["predicted_category"] for pred in mcq_predictions]
        mcq_true_categories = mcqs["curriculum_category"].tolist()
        mcq_accuracy = accuracy_score(mcq_true_categories, mcq_predicted_categories)
        
        # Evaluate open-ended questions
        open_predictions = self.batch_classify(open_ended["processed_stem"].tolist())
        open_predicted_categories = [pred["predicted_category"] for pred in open_predictions]
        open_true_categories = open_ended["curriculum_category"].tolist()
        open_accuracy = accuracy_score(open_true_categories, open_predicted_categories)
        
        return {
            "mcq_accuracy": mcq_accuracy,
            "open_ended_accuracy": open_accuracy,
            "format_gap": abs(mcq_accuracy - open_accuracy),
            "num_mcqs": len(mcqs),
            "num_open_ended": len(open_ended)
        }
    
    def evaluate_novel_term_adaptation(self, questions_df: pd.DataFrame, 
                                     novel_terms: List[str]) -> Dict:
        """
        Evaluate the model's ability to correctly classify questions containing novel terms.
        
        Args:
            questions_df: DataFrame of questions
            novel_terms: List of terms considered "novel" (e.g., emerging technologies)
        """
        # Find questions containing novel terms
        novel_questions = []
        
        for _, row in questions_df.iterrows():
            text = row["processed_stem"]
            if any(term.lower() in text.lower() for term in novel_terms):
                novel_questions.append(row)
        
        # If no novel questions found
        if not novel_questions:
            return {
                "novel_term_accuracy": None,
                "num_novel_questions": 0,
                "message": "No questions containing novel terms found"
            }
        
        # Convert to DataFrame
        novel_df = pd.DataFrame(novel_questions)
        
        # Classify novel questions
        predictions = self.batch_classify(novel_df["processed_stem"].tolist())
        predicted_categories = [pred["predicted_category"] for pred in predictions]
        true_categories = novel_df["curriculum_category"].tolist()
        
        # Calculate accuracy
        novel_accuracy = accuracy_score(true_categories, predicted_categories)
        
        # Calculate per-term accuracy
        per_term_accuracy = {}
        for term in novel_terms:
            term_questions = []
            term_true_categories = []
            term_predictions = []
            
            for _, row in novel_df.iterrows():
                if term.lower() in row["processed_stem"].lower():
                    term_questions.append(row)
                    term_true_categories.append(row["curriculum_category"])
                    term_predictions.append(predictions[novel_questions.index(row)]["predicted_category"])
            
            if term_questions:
                term_accuracy = accuracy_score(term_true_categories, term_predictions)
                per_term_accuracy[term] = {
                    "accuracy": term_accuracy,
                    "num_questions": len(term_questions)
                }
        
        return {
            "novel_term_accuracy": novel_accuracy,
            "num_novel_questions": len(novel_questions),
            "novel_terms_used": novel_terms,
            "per_term_accuracy": per_term_accuracy
        } 
    
    def _get_ensemble_prediction(self, model_predictions: Dict[str, Dict]) -> Dict:
        """
        Combine predictions from multiple models using the specified ensemble method.
        
        Args:
            model_predictions: Dictionary of model predictions
                              {model_name: {predicted_category, confidence, ...}}
                              
        Returns:
            Combined prediction result
        """
        available_models = list(model_predictions.keys())
        
        if not available_models:
            return {
                "predicted_category": "Unknown",
                "confidence": 0.0,
                "model_used": "none",
                "ensemble_method": None
            }
        
        if len(available_models) == 1:
            # Only one model available, use its prediction
            model = available_models[0]
            result = model_predictions[model].copy()
            result["model_used"] = model
            result["ensemble_method"] = None
            return result
        
        # Initialize result with combined predictions
        result = {
            "model_used": "ensemble",
            "ensemble_method": self.ensemble_method,
            "individual_predictions": model_predictions
        }
        
        if self.ensemble_method == "max_confidence":
            # Choose the model with highest confidence
            max_conf_model = max(available_models, key=lambda m: model_predictions[m]["confidence"])
            result["predicted_category"] = model_predictions[max_conf_model]["predicted_category"]
            result["confidence"] = model_predictions[max_conf_model]["confidence"]
            result["primary_model"] = max_conf_model
            
        elif self.ensemble_method == "weighted_vote":
            # Weighted voting across all models
            category_scores = {cat: 0.0 for cat in CURRICULUM_CATEGORIES}
            
            for model in available_models:
                weight = self.model_weights.get(model, 1.0 / len(available_models))
                pred = model_predictions[model]
                
                # Add weighted vote for the predicted category
                category_scores[pred["predicted_category"]] += weight * pred["confidence"]
                
                # If model provides probabilities for all categories, use them
                if "all_predictions" in pred and pred["all_predictions"]:
                    for cat, prob in pred["all_predictions"].items():
                        if cat != pred["predicted_category"]:  # Avoid double counting
                            category_scores[cat] += weight * prob
            
            # Find category with highest score
            max_category = max(category_scores.items(), key=lambda x: x[1])
            result["predicted_category"] = max_category[0]
            result["confidence"] = max_category[1]
            result["category_scores"] = category_scores
            
        elif self.ensemble_method == "cascade":
            # Cascade through models in order of preference
            # Order: transformer -> gpt4 -> baseline
            model_order = ["transformer", "gpt4_zero_shot", "baseline"]
            
            for model in model_order:
                if model in model_predictions:
                    pred = model_predictions[model]
                    if pred["confidence"] >= self.confidence_thresholds.get(model, DEFAULT_CONFIDENCE_THRESHOLD):
                        result["predicted_category"] = pred["predicted_category"]
                        result["confidence"] = pred["confidence"]
                        result["primary_model"] = model
                        break
            else:
                # If no model meets threshold, use the highest confidence prediction
                max_conf_model = max(available_models, key=lambda m: model_predictions[m]["confidence"])
                result["predicted_category"] = model_predictions[max_conf_model]["predicted_category"]
                result["confidence"] = model_predictions[max_conf_model]["confidence"]
                result["primary_model"] = max_conf_model
        
        return result
    
    def evaluate_all_models(self, test_questions: List[str], true_labels: List[str],
                          use_explanation: bool = True, use_ensemble: bool = True) -> Dict:
        """
        Evaluate all available models on the same test set.
        
        Args:
            test_questions: List of test questions
            true_labels: List of true labels
            use_explanation: Whether to use explanation text if available
            use_ensemble: Whether to use ensemble methods
            
        Returns:
            Dictionary of evaluation results for each model
        """
        results = {
            "metrics": {},
            "predictions": {}
        }
        
        # Evaluate baseline model if available
        if self.baseline_model:
            baseline_preds = self.baseline_model.predict(test_questions)
            baseline_probs = self.baseline_model.predict_proba(test_questions)
            
            # Calculate metrics
            accuracy = accuracy_score(true_labels, baseline_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, baseline_preds, average='weighted'
            )
            
            results["metrics"]["baseline"] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
            
            results["predictions"]["baseline"] = {
                "predicted_labels": baseline_preds,
                "probabilities": baseline_probs
            }
        
        # Evaluate transformer model if available
        if self.transformer_model:
            transformer_preds, transformer_probs = self.transformer_model.predict(test_questions)
            
            # Calculate metrics
            accuracy = accuracy_score(true_labels, transformer_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, transformer_preds, average='weighted'
            )
            
            results["metrics"]["transformer"] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
            
            results["predictions"]["transformer"] = {
                "predicted_labels": transformer_preds,
                "probabilities": transformer_probs
            }
        
        # Evaluate GPT-4 model if available
        if self.gpt4_model:
            gpt4_results = self.gpt4_model.classify(test_questions)
            gpt4_preds = [result["predicted_category"] for result in gpt4_results]
            
            # Calculate metrics
            accuracy = accuracy_score(true_labels, gpt4_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, gpt4_preds, average='weighted'
            )
            
            results["metrics"]["gpt4_zero_shot"] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
            
            results["predictions"]["gpt4_zero_shot"] = {
                "predicted_labels": gpt4_preds,
                "results": gpt4_results
            }
        
        # Evaluate ensemble if requested and multiple models are available
        available_models = sum([
            1 if self.baseline_model else 0,
            1 if self.transformer_model else 0,
            1 if self.gpt4_model else 0
        ])
        
        if use_ensemble and available_models > 1:
            ensemble_results = self.batch_classify(
                test_questions, 
                use_explanation=use_explanation,
                use_ensemble=True
            )
            
            ensemble_preds = [result["predicted_category"] for result in ensemble_results]
            
            # Calculate metrics
            accuracy = accuracy_score(true_labels, ensemble_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, ensemble_preds, average='weighted'
            )
            
            results["metrics"]["ensemble"] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "ensemble_method": self.ensemble_method
            }
            
            results["predictions"]["ensemble"] = {
                "predicted_labels": ensemble_preds,
                "results": ensemble_results
            }
        
        # Save performance history for ensemble
        if "ensemble" in results["metrics"]:
            self.performance_history["ensemble"].append({
                "timestamp": datetime.now().isoformat(),
                "metrics": results["metrics"]["ensemble"],
                "ensemble_config": {
                    "method": self.ensemble_method,
                    "weights": self.model_weights,
                    "thresholds": self.confidence_thresholds
                }
            })
        
        return results
    
    def plot_confusion_matrices(self, evaluation_results: Dict, figsize: Tuple[int, int] = (20, 15)):
        """
        Plot confusion matrices for all models in the evaluation results.
        
        Args:
            evaluation_results: Results from evaluate_all_models
            figsize: Figure size for the plot
        """
        # Get all available models
        available_models = list(evaluation_results["predictions"].keys())
        
        if not available_models:
            print("No models available for plotting confusion matrices.")
            return
        
        # Create subplots
        fig, axes = plt.subplots(1, len(available_models), figsize=figsize)
        
        # Handle case with only one model
        if len(available_models) == 1:
            axes = [axes]
        
        for i, model_name in enumerate(available_models):
            # Get predictions and true labels
            if model_name == "ensemble":
                true_labels = [result["true_label"] for result in evaluation_results["predictions"][model_name]["results"]]
                pred_labels = evaluation_results["predictions"][model_name]["predicted_labels"]
            else:
                true_labels = [result["true_label"] for result in evaluation_results["predictions"][model_name]["results"]]
                pred_labels = evaluation_results["predictions"][model_name]["predicted_labels"]
            
            # Calculate confusion matrix
            cm = confusion_matrix(true_labels, pred_labels, labels=CURRICULUM_CATEGORIES)
            
            # Plot confusion matrix
            sns.heatmap(
                cm, 
                annot=True, 
                fmt="d", 
                cmap="Blues",
                xticklabels=CURRICULUM_CATEGORIES,
                yticklabels=CURRICULUM_CATEGORIES,
                ax=axes[i]
            )
            
            axes[i].set_title(f"{model_name.capitalize()} Confusion Matrix")
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("True")
            plt.setp(axes[i].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.save_dir, f"confusion_matrices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(save_path)
        print(f"Confusion matrices saved to {save_path}")
        
        return fig
    
    def plot_performance_comparison(self, evaluation_results: Dict, metric: str = "accuracy", figsize: Tuple[int, int] = (10, 6)):
        """
        Plot performance comparison across models.
        
        Args:
            evaluation_results: Results from evaluate_all_models
            metric: Metric to compare ("accuracy", "precision", "recall", "f1")
            figsize: Figure size for the plot
        """
        # Get metrics for all models
        models = []
        values = []
        
        for model_name, metrics in evaluation_results["metrics"].items():
            if metric in metrics:
                models.append(model_name.capitalize())
                values.append(metrics[metric])
        
        if not models:
            print(f"No models available with metric '{metric}'.")
            return
        
        # Create bar plot
        plt.figure(figsize=figsize)
        bars = plt.bar(models, values)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom"
            )
        
        plt.title(f"Model Performance Comparison ({metric.capitalize()})")
        plt.ylabel(metric.capitalize())
        plt.ylim(0, 1.1)  # Assuming metrics are between 0 and 1
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Save figure
        save_path = os.path.join(self.save_dir, f"performance_comparison_{metric}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(save_path)
        print(f"Performance comparison saved to {save_path}")
        
        return plt.gcf()
    
    def analyze_misclassifications(self, evaluation_results: Dict, model_name: str, 
                                 test_questions: List[str], true_labels: List[str],
                                 n_examples: int = 5) -> Dict:
        """
        Analyze misclassifications for a specific model.
        
        Args:
            evaluation_results: Results from evaluate_all_models
            model_name: Name of the model to analyze
            test_questions: List of test questions
            true_labels: List of true labels
            n_examples: Number of examples to include in the report
            
        Returns:
            Dictionary with misclassification analysis
        """
        if model_name not in evaluation_results["predictions"]:
            return {"error": f"Model '{model_name}' not found in evaluation results."}
        
        # Get predictions
        if model_name == "ensemble":
            pred_labels = evaluation_results["predictions"][model_name]["predicted_labels"]
        else:
            pred_labels = evaluation_results["predictions"][model_name]["predicted_labels"]
        
        # Find misclassifications
        misclassified_indices = [
            i for i, (true, pred) in enumerate(zip(true_labels, pred_labels))
            if true != pred
        ]
        
        # Prepare analysis
        analysis = {
            "model_name": model_name,
            "total_examples": len(test_questions),
            "misclassified_count": len(misclassified_indices),
            "misclassification_rate": len(misclassified_indices) / len(test_questions),
            "examples": []
        }
        
        # Add examples
        for idx in misclassified_indices[:n_examples]:
            example = {
                "question": test_questions[idx],
                "true_label": true_labels[idx],
                "predicted_label": pred_labels[idx]
            }
            
            # Add model-specific details
            if model_name == "ensemble":
                result = evaluation_results["predictions"][model_name]["results"][idx]
                example["confidence"] = result["confidence"]
                example["ensemble_method"] = result["ensemble_method"]
                example["individual_predictions"] = result.get("individual_predictions", {})
            elif model_name == "gpt4_zero_shot":
                result = evaluation_results["predictions"][model_name]["results"][idx]
                example["confidence"] = result["confidence"]
                example["reasoning"] = result.get("reasoning", "")
            else:
                # For baseline and transformer
                if "probabilities" in evaluation_results["predictions"][model_name]:
                    probs = evaluation_results["predictions"][model_name]["probabilities"][idx]
                    if isinstance(probs, list):
                        example["confidence"] = max(probs)
                    else:
                        example["confidence"] = probs.max()
            
            analysis["examples"].append(example)
        
        # Calculate confusion patterns
        confusion_patterns = {}
        for idx in misclassified_indices:
            true = true_labels[idx]
            pred = pred_labels[idx]
            pattern = f"{true} → {pred}"
            
            if pattern not in confusion_patterns:
                confusion_patterns[pattern] = 0
            confusion_patterns[pattern] += 1
        
        # Sort patterns by frequency
        sorted_patterns = sorted(confusion_patterns.items(), key=lambda x: x[1], reverse=True)
        analysis["confusion_patterns"] = {pattern: count for pattern, count in sorted_patterns}
        
        return analysis
    
    def save_model(self, filepath: str = None):
        """
        Save the classifier to a file.
        
        Args:
            filepath: Path to save the model (default: save_dir/exam_classifier_TIMESTAMP.pkl)
        """
        if filepath is None:
            filepath = os.path.join(self.save_dir, f"exam_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save models separately if they exist
        model_paths = {}
        
        if self.baseline_model:
            baseline_path = os.path.join(self.save_dir, "baseline_model.pkl")
            self.baseline_model.save_model(baseline_path)
            model_paths["baseline"] = baseline_path
        
        if self.transformer_model:
            transformer_path = os.path.join(self.save_dir, "transformer_model")
            self.transformer_model.save_model(transformer_path)
            model_paths["transformer"] = transformer_path
        
        # Save configuration and paths
        config = {
            "model_paths": model_paths,
            "ensemble_method": self.ensemble_method,
            "model_weights": self.model_weights,
            "confidence_thresholds": self.confidence_thresholds,
            "performance_history": self.performance_history,
            "save_dir": self.save_dir
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(config, f)
        
        print(f"Model saved to {filepath}")
        return filepath
    
    @classmethod
    def load_model(cls, filepath: str):
        """
        Load a classifier from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded ExamQuestionClassifier
        """
        with open(filepath, "rb") as f:
            config = pickle.load(f)
        
        # Create instance with configuration
        classifier = cls(
            ensemble_method=config["ensemble_method"],
            model_weights=config["model_weights"],
            confidence_thresholds=config["confidence_thresholds"],
            save_dir=config["save_dir"]
        )
        
        # Load models if paths exist
        if "baseline" in config["model_paths"]:
            baseline_path = config["model_paths"]["baseline"]
            classifier.baseline_model = BaselineClassifier.load_model(baseline_path)
        
        if "transformer" in config["model_paths"]:
            transformer_path = config["model_paths"]["transformer"]
            classifier.transformer_model = TransformerClassifier.load_model(transformer_path)
        
        # Load performance history
        classifier.performance_history = config["performance_history"]
        
        return classifier