"""
Main application class for Llama Zero-Shot Classification
"""

import os
import json
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple

class LlamaClassifierApp:
    """
    Application for managing and running the Llama Zero-Shot Classifier
    for medical physics exam questions.
    """
    
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-2-7b-chat-hf",
                 curriculum_path: Optional[str] = None,
                 output_dir: str = "results",
                 hf_token: Optional[str] = None):
        """
        Initialize the classifier application
        
        Args:
            model_name: Name of the Hugging Face model to use
            curriculum_path: Path to the curriculum JSON file
            output_dir: Directory to save outputs
            hf_token: Hugging Face API token
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.curriculum_categories = {}
        
        # Load curriculum if provided
        if curriculum_path:
            self.load_curriculum(curriculum_path)
            
        # Initialize classifier
        self.classifier = LlamaZeroShotClassifier(
            model_name=model_name,
            curriculum_categories=self.curriculum_categories,
            hf_token=hf_token
        )
        
        print(f"LlamaClassifierApp initialized with model: {model_name}")
        
    def load_curriculum(self, curriculum_path: str) -> Dict:
        """
        Load curriculum categories from a JSON file
        
        Args:
            curriculum_path: Path to the curriculum JSON file
            
        Returns:
            Dictionary of curriculum categories
        """
        try:
            with open(curriculum_path, 'r') as f:
                self.curriculum_categories = json.load(f)
        
            
            # Reinitialize classifier with new curriculum if it exists
            if hasattr(self, 'classifier'):
                self.classifier.curriculum_categories = self.curriculum_categories
                self.classifier.category_list = list(self.curriculum_categories.keys())
                
            return self.curriculum_categories
        except Exception as e:
           
            raise
            
    def load_questions(self, questions_path: str) -> List[Dict]:
        """
        Load questions from a JSON file
        
        Args:
            questions_path: Path to the questions JSON file
            
        Returns:
            List of question dictionaries
        """
        try:
            with open(questions_path, 'r') as f:
                questions_data = json.load(f)
         
            return questions_data
        except Exception as e:
     
            raise
    
    def classify_question(self, question: str) -> Tuple[str, float]:
        """
        Classify a single question
        
        Args:
            question: The question text to classify
            
        Returns:
            Tuple of (category, confidence)
        """
        if not self.curriculum_categories:
         
            return "Unknown", 0.0
            
        return self.classifier.classify_question(question)
    
    def run_batch_classification(self, 
                              questions_path: str, 
                              batch_size: int = 8, 
                              save_results: bool = True) -> Dict:
        """
        Run batch classification on a set of questions
        
        Args:
            questions_path: Path to the questions JSON file
            batch_size: Number of questions to process in parallel
            save_results: Whether to save results to file
            
        Returns:
            Dictionary with evaluation results
        """
        # Load questions
        questions_data = self.load_questions(questions_path)
        questions = [q["question_text"] for q in questions_data]
        
        # Extract true labels if available
        true_labels = [q.get("category", "") for q in questions_data]
        has_true_labels = all(label != "" for label in true_labels)
        
        if has_true_labels:
         
            results = self.classifier.evaluate(
                questions=questions,
                true_labels=true_labels,
                output_file=str(self.output_dir / "evaluation_results.json") if save_results else None
            )
            
            # Visualize results if true labels are available
            if save_results:
                self.visualize_results(results)
                
            return results
        else:
      
            predictions = self.classifier.classify_batch(questions, batch_size)
            
            if save_results:
                self.classifier.save_results(
                    questions=questions,
                    predictions=predictions,
                    output_file=str(self.output_dir / "classification_results.json")
                )
                
            return {
                "predictions": [
                    {"question": q, "predicted": p[0], "confidence": p[1]}
                    for q, p in zip(questions, predictions)
                ]
            }
    
    def visualize_results(self, results: Dict) -> None:
        """
        Create visualizations of classification results
        
        Args:
            results: Dictionary with evaluation results
        """
        # Create confusion matrix visualization
        plt.figure(figsize=(12, 10))
        confusion_matrix = results["confusion_matrix"]
        categories = results["categories"]
        
        # Plot confusion matrix
        sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", 
                  xticklabels=categories, yticklabels=categories)
        plt.title("Confusion Matrix")
        plt.ylabel("True Category")
        plt.xlabel("Predicted Category")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(str(self.output_dir / "confusion_matrix.png"))
        plt.close()
        
        # Create category distribution visualization
        predictions = [item["predicted"] for item in results["detailed_results"]]
        true_labels = [item["true"] for item in results["detailed_results"]]
        
        # Plot category distribution
        plt.figure(figsize=(14, 6))
        
        # True distribution
        plt.subplot(1, 2, 1)
        true_counts = pd.Series(true_labels).value_counts()
        true_counts.plot(kind="bar", color="green", alpha=0.7)
        plt.title("True Category Distribution")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        # Predicted distribution
        plt.subplot(1, 2, 2)
        pred_counts = pd.Series(predictions).value_counts()
        pred_counts.plot(kind="bar", color="blue", alpha=0.7)
        plt.title("Predicted Category Distribution")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        plt.savefig(str(self.output_dir / "category_distribution.png"))
        plt.close()
        
        # Calculate per-category metrics
        metrics = []
        for category in categories:
            true_positives = sum(1 for t, p in zip(true_labels, predictions) 
                              if t == category and p == category)
            false_positives = sum(1 for t, p in zip(true_labels, predictions) 
                               if t != category and p == category)
            false_negatives = sum(1 for t, p in zip(true_labels, predictions) 
                               if t == category and p != category)
            
            # Calculate precision, recall, and F1 score
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics.append({
                "category": category,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": true_counts.get(category, 0)
            })
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(str(self.output_dir / "category_metrics.csv"), index=False)
        
        # Plot precision, recall, F1 scores
        plt.figure(figsize=(14, 8))
        metrics_df = metrics_df.sort_values("support", ascending=False)
        
        bar_width = 0.25
        indices = range(len(metrics_df))
        
        plt.bar([i for i in indices], metrics_df["precision"], bar_width, 
              label="Precision", color="blue", alpha=0.7)
        plt.bar([i + bar_width for i in indices], metrics_df["recall"], bar_width,
              label="Recall", color="green", alpha=0.7)
        plt.bar([i + 2*bar_width for i in indices], metrics_df["f1"], bar_width,
              label="F1 Score", color="red", alpha=0.7)
        
        plt.xlabel("Category")
        plt.ylabel("Score")
        plt.title("Precision, Recall, and F1 Score by Category")
        plt.xticks([i + bar_width for i in indices], metrics_df["category"], rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(self.output_dir / "category_metrics.png"))
        plt.close()
        

        
    def export_results_to_csv(self, results: Dict, output_file: str) -> None:
        """
        Export classification results to CSV format
        
        Args:
            results: Dictionary with evaluation or classification results
            output_file: Path to output CSV file
        """
        if "detailed_results" in results:
            # Evaluation results
            df = pd.DataFrame(results["detailed_results"])
        else:
            # Classification results
            df = pd.DataFrame(results["predictions"])
            
        df.to_csv(output_file, index=False)
    
        
    def interactive_mode(self) -> None:
        """Run an interactive classification session"""
        print("\n===== Llama Zero-Shot Classifier Interactive Mode =====")
        print(f"Model: {self.model_name}")
        print(f"Categories: {len(self.curriculum_categories)}")
        print("Enter 'q' to quit")
        
        while True:
            question = input("\nEnter question to classify: ")
            if question.lower() == 'q':
                break
                
            category, confidence = self.classify_question(question)
            print(f"\nClassified as: {category}")
            print(f"Confidence: {confidence:.3f}")


def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description="Llama Zero-Shot Classification Application")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                      help="HuggingFace model name")
    parser.add_argument("--curriculum", type=str, required=True,
                      help="Path to curriculum JSON file")
    parser.add_argument("--questions", type=str, default=None,
                      help="Path to questions JSON file (for batch processing)")
    parser.add_argument("--output", type=str, default="results",
                      help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=8,
                      help="Batch size for processing")
    parser.add_argument("--hf_token", type=str, default=None,
                      help="HuggingFace API token")
    parser.add_argument("--interactive", action="store_true",
                      help="Run in interactive mode")
    parser.add_argument("--export_csv", action="store_true",
                      help="Export results to CSV format")
    
    args = parser.parse_args()
    
    # Initialize the application
    app = LlamaClassifierApp(
        model_name=args.model,
        curriculum_path=args.curriculum,
        output_dir=args.output,
        hf_token=args.hf_token
    )
    
    # Run in interactive mode if requested
    if args.interactive:
        app.interactive_mode()
    # Run batch classification if questions file is provided
    elif args.questions:
        results = app.run_batch_classification(
            questions_path=args.questions,
            batch_size=args.batch_size,
            save_results=True
        )
        
        # Export to CSV if requested
        if args.export_csv:
            app.export_results_to_csv(
                results,
                output_file=str(Path(args.output) / "results.csv")
            )
    else:
        print("Either --questions or --interactive must be specified")
        parser.print_help()
        

if __name__ == "__main__":
    main()