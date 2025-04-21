"""
Implements transformer-based classification for medical physics exam questions.
"""

import os
import torch
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report

from src.config.constants import (
    CURRICULUM_CATEGORIES,
    DEFAULT_MODEL_NAME,
    DEFAULT_MAX_LENGTH,
    MODEL_OUTPUT_DIR
)

class TransformerClassifier:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        """Initialize the transformer classifier."""
        self.model_name = model_name
        self.num_labels = len(CURRICULUM_CATEGORIES)
        self.max_length = DEFAULT_MAX_LENGTH
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=self.num_labels
        )
        
        # Set up label mappings
        self.id2label = {i: label for i, label in enumerate(CURRICULUM_CATEGORIES)}
        self.label2id = {label: i for i, label in enumerate(CURRICULUM_CATEGORIES)}
    
    def prepare_data(self, texts: List[str], labels: List[str] = None):
        """Prepare data for training/inference."""
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        if labels:
            label_ids = [self.label2id[label] for label in labels]
            encodings["labels"] = torch.tensor(label_ids)
        
        return encodings
    
    def train(self, train_texts: List[str], train_labels: List[str],
              eval_texts: List[str] = None, eval_labels: List[str] = None) -> Dict:
        """Train the transformer model."""
        # Prepare datasets
        train_encodings = self.prepare_data(train_texts, train_labels)
        
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings):
                self.encodings = encodings
            def __getitem__(self, idx):
                return {key: val[idx] for key, val in self.encodings.items()}
            def __len__(self):
                return len(self.encodings["input_ids"])
        
        train_dataset = Dataset(train_encodings)
        eval_dataset = None
        if eval_texts and eval_labels:
            eval_encodings = self.prepare_data(eval_texts, eval_labels)
            eval_dataset = Dataset(eval_encodings)
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=MODEL_OUTPUT_DIR,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False
        )
        
        # Train model
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        trainer.train()
        
        # Save model
        self.model.save_pretrained(MODEL_OUTPUT_DIR)
        self.tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
        
        return {"status": "success"}
    
    def predict(self, texts: List[str]) -> Tuple[List[str], List[float]]:
        """Predict categories for texts."""
        encodings = self.prepare_data(texts)
        trainer = Trainer(model=self.model)
        
        predictions = trainer.predict(Dataset(encodings)).predictions
        predicted_ids = predictions.argmax(axis=1)
        
        # Get labels and confidences
        labels = [self.id2label[id] for id in predicted_ids]
        confidences = predictions.max(axis=1)
        
        return labels, confidences
    
    def evaluate(self, texts: List[str], true_labels: List[str]) -> Dict:
        """Evaluate the model."""
        predicted_labels, confidences = self.predict(texts)
        
        return {
            "accuracy": accuracy_score(true_labels, predicted_labels),
            "classification_report": classification_report(true_labels, predicted_labels),
            "predictions": list(zip(texts, predicted_labels, true_labels, confidences))
        }
    
    @classmethod
    def load(cls, model_dir: str) -> 'TransformerClassifier':
        """Load a trained model."""
        instance = cls()
        instance.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        instance.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        return instance 
    