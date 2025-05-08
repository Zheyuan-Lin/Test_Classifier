import os
import torch
from typing import List, Dict, Tuple, Optional, Union
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import Dataset as TorchDataset


class transformerClassifier:
    """Transformer-based classifier for physics exam questions categorization."""
    
    def __init__(
        self, 
        model_name: str = "allenai/scibert_scivocab_uncased",
        categories: List[str] = None,
        max_length: int = 128,
        output_dir: str = "./physics_classifier"
    ):
        """Initialize the physics exam classifier.
        
        Args:
            model_name: Pretrained transformer model to use
            categories: List of curriculum categories for classification
            max_length: Maximum sequence length for tokenization
            output_dir: Directory to save model outputs
        """
        self.model_name = model_name
        self.max_length = max_length
        self.output_dir = output_dir
        
        # Use provided categories or default ones
        self.categories = categories or[
    "Fundamental Physics",
    "Atomic and Nuclear Structure",
    "Production of Kilovoltage X-ray Beams",
    "Production of Megavoltage Radiation Beams",
    "Radiation Interactions",
    "Radiation Quantities and Units",
    "Radiation Measurement and Calibration",
    "Photon Beam Characteristics and Dosimetry",
    "Electron Beam Characteristics and Dosimetry",
    "Intensity Modulated Radiation Therapy (IMRT)",
    "Prescribing, Reporting, and Evaluating Radiotherapy Treatment Plans",
    "Imaging Fundamentals",
    "Simulation, Motion Management and Treatment Verification",
    "Clinical Brachytherapy",
    "Brachytherapy QA",
    "Advanced Treatment Planning and Special Procedures",
    "Particle Therapy",
    "Stereotactic Radiosurgery / Stereotactic Body Radiotherapy (SRS/SBRT)",
    "Quality Assurance in Radiation Oncology",
]
        self.num_labels = len(self.categories)
        
        # Initialize mappings
        self.id2label = {i: label for i, label in enumerate(self.categories)}
        self.label2id = {label: i for i, label in enumerate(self.categories)}
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
    
    class _Dataset(TorchDataset):
        """Internal dataset class for handling encodings."""
        def __init__(self, encodings):
            self.encodings = encodings
            
        def __getitem__(self, idx):
            return {key: val[idx] for key, val in self.encodings.items()}
            
        def __len__(self):
            return len(self.encodings["input_ids"])
    
    def _encode_data(self, texts: List[str], labels: Optional[List[str]] = None) -> Dict:
        """Tokenize and encode the data.
        
        Args:
            texts: List of question texts
            labels: Optional list of category labels
            
        Returns:
            Dictionary of encoded data
        """
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
    
    def train(
        self,
        train_texts: List[str],
        train_labels: List[str],
        eval_texts: Optional[List[str]] = None,
        eval_labels: Optional[List[str]] = None,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5
    ) -> Dict:
        """Train the classifier on physics exam questions.
        
        Args:
            train_texts: Training question texts
            train_labels: Training category labels
            eval_texts: Evaluation question texts
            eval_labels: Evaluation category labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for training
            
        Returns:
            Dictionary containing training results
        """
        # Create datasets
        train_encodings = self._encode_data(train_texts, train_labels)
        train_dataset = self._Dataset(train_encodings)
        
        eval_dataset = None
        if eval_texts and eval_labels:
            eval_encodings = self._encode_data(eval_texts, eval_labels)
            eval_dataset = self._Dataset(eval_encodings)
        
        # Configure training
        os.makedirs(self.output_dir, exist_ok=True)
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=bool(eval_dataset),
            save_total_limit=2,  # Only keep 2 checkpoints
            report_to="none"  # Disable wandb/tensorboard reporting
        )
        
        # Train model
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        train_result = trainer.train()
        
        # Save model
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Return basic metrics
        result = {"status": "success", "training_loss": train_result.training_loss}
        
        if eval_dataset:
            eval_result = trainer.evaluate()
            result["eval_loss"] = eval_result["eval_loss"]
            
        return result
    
    def predict(self, texts: List[str]) -> Tuple[List[str], List[float]]:
        """Predict categories for physics exam questions.
        
        Args:
            texts: List of question texts
            
        Returns:
            Tuple of (predicted_categories, confidence_scores)
        """
        # Encode data
        encodings = self._encode_data(texts)
        dataset = self._Dataset(encodings)
        
        # Make predictions
        trainer = Trainer(model=self.model)
        raw_predictions = trainer.predict(dataset).predictions
        
        # Get predictions and confidences
        predicted_ids = raw_predictions.argmax(axis=1)
        labels = [self.id2label[int(id_)] for id_ in predicted_ids]
        
        # Get confidence scores using softmax
        scores = torch.nn.functional.softmax(torch.tensor(raw_predictions), dim=1)
        confidences = scores.max(dim=1).values.tolist()
        
        return labels, confidences
    
    def evaluate(self, texts: List[str], true_labels: List[str]) -> Dict:
        """Evaluate the model on physics exam questions.
        
        Args:
            texts: List of question texts
            true_labels: List of true category labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        predicted_labels, confidences = self.predict(texts)
        
        return {
            "accuracy": accuracy_score(true_labels, predicted_labels),
            "classification_report": classification_report(
                true_labels, 
                predicted_labels, 
                target_names=self.categories
            ),
            "predictions": list(zip(texts, predicted_labels, true_labels, confidences))
        }
    
    @classmethod
    def load(cls, model_dir: str, categories: Optional[List[str]] = None) -> 'transformerClassifier':
        """Load a pretrained physics exam classifier.
        
        Args:
            model_dir: Directory containing saved model
            categories: Optional list of curriculum categories
            
        Returns:
            Loaded PhysicsExamClassifier instance
        """
        # Create instance
        instance = cls(model_name=model_dir, categories=categories)
        
        # Load model and tokenizer
        instance.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        instance.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # If categories not provided, try to load from model config
        if not categories and hasattr(instance.model.config, 'id2label'):
            instance.id2label = instance.model.config.id2label
            instance.label2id = instance.model.config.label2id
            instance.categories = list(instance.label2id.keys())
            instance.num_labels = len(instance.categories)
        
        return instance
    
    def save(self, output_dir: Optional[str] = None) -> str:
        """Save the model to disk.
        
        Args:
            output_dir: Optional directory to save the model to
            
        Returns:
            Path to saved model
        """
        save_dir = output_dir or self.output_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        return save_dir

def main():
    """Main function to demonstrate the usage of the transformer classifier."""
    import json
    import os
    from sklearn.model_selection import train_test_split
    
    # Initialize the classifier
    classifier = transformerClassifier(
        model_name="allenai/scibert_scivocab_uncased",
        max_length=128,
        output_dir="./transformer_models"
    )
    
    # Find JSON file
    json_files = [f for f in os.listdir('.') if f.endswith('.json')]
    if json_files:
        json_path = json_files[0]
    else:
        # Use default path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, '..', '..', 'data', 'processed', 'synthetic.json')
    
    print(f"Loading data from: {json_path}")
    
    # Load and prepare data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract texts and labels
    texts = [item['question_text'] for item in data['dataset']]
    labels = [item['category'] for item in data['dataset']]
    
    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training set size: {len(train_texts)}")
    print(f"Test set size: {len(test_texts)}")
    
    # Train the model
    print("\nTraining transformer model...")
    training_results = classifier.train(
        train_texts=train_texts,
        train_labels=train_labels,
        eval_texts=test_texts,
        eval_labels=test_labels,
        epochs=3,
        batch_size=8,
        learning_rate=5e-5
    )
    
    # Evaluate the model
    print("\nEvaluating transformer model...")
    eval_results = classifier.evaluate(test_texts, test_labels)
    
    # Print results
    print("\n===== RESULTS SUMMARY =====")
    print(f"Training Loss: {training_results['training_loss']:.3f}")
    if 'eval_loss' in training_results:
        print(f"Evaluation Loss: {training_results['eval_loss']:.3f}")
    print(f"Test Accuracy: {eval_results['accuracy']:.3f}")
    print("\nDetailed Classification Report:")
    print(eval_results['classification_report'])
    
    # Save the model
    save_path = classifier.save("./transformer_models")
    print(f"\nModel saved to: {save_path}")
    
    # Example prediction
    print("\nExample prediction:")
    sample_text = "Calculate the force required to accelerate a 2kg mass at 5m/s²"
    predicted_label, confidence = classifier.predict([sample_text])
    print(f"Input text: {sample_text}")
    print(f"Predicted category: {predicted_label[0]}")
    print(f"Confidence: {confidence[0]:.3f}")

if __name__ == "__main__":
    main()