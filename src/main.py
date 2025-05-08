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
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

from utils.logger import get_logger
from preprocessing.text_preprocessor import TextPreprocessor
from models.unsupervised_topic_model import UnsupervisedTopicModel
from models.gpt_zero_shot_classifier import LlamaZeroShotClassifier
from preprocessing.pdf_question_extractor import PDFQuestionExtractor
from config.constants import RAW_DATA_DIR, CURRICULUM_CATEGORIES

logger = get_logger(__name__)

class ExamQuestionPipeline:
    def __init__(self, 
                 data_dir: str = "data",
                 model_dir: str = "models",
                 results_dir: str = "results",
                 test_size: float = 0.2,
                 val_size: float = 0.2,
                 random_state: int = 42,
                 classifier_config: Dict = None,
                 hf_token: str = None):
        """Initialize the pipeline."""
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.results_dir = results_dir
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        # Initialize components using existing classes
        self.extractor = PDFQuestionExtractor()
        self.preprocessor = TextPreprocessor()
        self.llama_classifier = LlamaZeroShotClassifier(
            curriculum_categories=CURRICULUM_CATEGORIES,
            hf_token=hf_token
        )

    def run_topic_modeling(self, questions: List[str], n_topics: int = 10, model_type: str = "lda") -> Dict:
        """Run topic modeling on the questions and save results."""
        logger.info(f"Running topic modeling with {n_topics} topics using {model_type}...")
        
        # Initialize topic model
        topic_model = UnsupervisedTopicModel(
            model_type=model_type,
            n_topics=n_topics
        )
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Transform questions to TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform(questions)
        feature_names = vectorizer.get_feature_names_out()
        
        # Fit the model
        topic_model.fit(tfidf_matrix)
        
        # Get topic distributions
        topic_distributions = topic_model.transform(tfidf_matrix)
        
        # Get top words for each topic
        topics = topic_model.get_topics(feature_names, n_top_words=10)
        
        # Map topics to curriculum categories
        topic_category_mapping = self._map_topics_to_categories(topics, vectorizer)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(self.results_dir, f"topic_modeling_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save topics to JSON
        topics_file = os.path.join(results_dir, "topics.json")
        with open(topics_file, 'w') as f:
            json.dump({str(k): v for k, v in topics.items()}, f, indent=2)
        
        # Save topic-category mapping
        mapping_file = os.path.join(results_dir, "topic_category_mapping.json")
        with open(mapping_file, 'w') as f:
            json.dump(topic_category_mapping, f, indent=2)
        
        # Save topic distributions
        distributions_file = os.path.join(results_dir, "topic_distributions.npy")
        np.save(distributions_file, topic_distributions)
        
        # Create visualization
        self._visualize_topics(topics, results_dir)
        self._visualize_topic_category_mapping(topic_category_mapping, results_dir)
        
        logger.info(f"Topic modeling results saved to {results_dir}")
        
        return {
            'model': topic_model,
            'vectorizer': vectorizer,
            'topics': topics,
            'topic_distributions': topic_distributions,
            'topic_category_mapping': topic_category_mapping
        }

    def run_llama_classification(self, questions: List[str]) -> Dict:
        """Run Llama 2 zero-shot classification on the questions."""
        logger.info("Running Llama 2 zero-shot classification...")
        
        # Classify questions
        predictions = self.llama_classifier.classify_batch(questions)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(self.results_dir, f"llama_classification_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save predictions
        results_file = os.path.join(results_dir, "predictions.json")
        self.llama_classifier.save_results(questions, predictions, results_file)
        
        # Create visualization
        #self._visualize_llama_predictions(questions, predictions, results_dir)
        
        logger.info(f"Llama 2 classification results saved to {results_dir}")
        
        return {
            'predictions': predictions,
            'results_dir': results_dir
        }

    def _visualize_llama_predictions(self, 
                                   questions: List[str], 
                                   predictions: List[Tuple[str, float]], 
                                   output_dir: str):
        """Create visualizations for Llama 2 predictions."""
        # Create category distribution plot
        categories = [p[0] for p in predictions]
        category_counts = pd.Series(categories).value_counts()
        
        plt.figure(figsize=(12, 6))
        category_counts.plot(kind='bar')
        plt.title('Distribution of Questions Across Categories')
        plt.xlabel('Category')
        plt.ylabel('Number of Questions')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(output_dir, "category_distribution.png")
        plt.savefig(plot_file)
        plt.close()
        
        # Create category confidence plot
        confidences = [p[1] for p in predictions]
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=20)
        plt.title('Distribution of Classification Confidence')
        plt.xlabel('Confidence')
        plt.ylabel('Number of Questions')
        plt.tight_layout()
        
        # Save plot
        conf_file = os.path.join(output_dir, "confidence_distribution.png")
        plt.savefig(conf_file)
        plt.close()

def main():
    # Get Hugging Face token from environment or user input
    hf_token = os.environ.get("HUGGING_FACE_TOKEN")
    if not hf_token:
        hf_token = input("Please enter your Hugging Face token: ")
    
    # 1. Load tokenized questions
    with open('/Users/soukasumi/Desktop/GLLP/data/processed/tokenized_questions.json', 'r') as f:
        data = json.load(f)
        questions = data['questions']
    
    # 2. Initialize pipeline with Hugging Face token
    pipeline = ExamQuestionPipeline(hf_token=hf_token)
    
    # 3. Run Llama 2 classification
    llama_results = pipeline.run_llama_classification(questions)
    
    logger.info("Llama 2 classification completed successfully!")

if __name__ == "__main__":
    main() 