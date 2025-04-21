"""
Configuration management utility for the exam classification pipeline.
"""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    model_name: str
    max_length: int
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    use_explanation: bool = False

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    test_size: float
    random_state: int
    class_weights: Optional[list] = None
    output_dir: str = "./transformer_model"

@dataclass
class ClassificationConfig:
    """Configuration for classification parameters."""
    confidence_threshold: float
    use_explanation: bool = True

class ConfigManager:
    """Manages configuration for the exam classification pipeline."""
    
    def __init__(self, config_dir: str = "./config"):
        """Initialize the configuration manager."""
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)
        
        # Default configurations
        self.model_config = ModelConfig(
            model_name="allenai/scibert_scivocab_uncased",
            max_length=512,
            batch_size=8,
            num_epochs=3,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.1
        )
        
        self.training_config = TrainingConfig(
            test_size=0.2,
            random_state=42
        )
        
        self.classification_config = ClassificationConfig(
            confidence_threshold=0.7
        )
    
    def save_config(self, name: str, config: Dict[str, Any]):
        """Save configuration to a JSON file."""
        config_path = os.path.join(self.config_dir, f"{name}.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
    
    def load_config(self, name: str) -> Dict[str, Any]:
        """Load configuration from a JSON file."""
        config_path = os.path.join(self.config_dir, f"{name}.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, "r") as f:
            return json.load(f)
    
    def update_model_config(self, **kwargs):
        """Update model configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.model_config, key):
                setattr(self.model_config, key, value)
    
    def update_training_config(self, **kwargs):
        """Update training configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.training_config, key):
                setattr(self.training_config, key, value)
    
    def update_classification_config(self, **kwargs):
        """Update classification configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.classification_config, key):
                setattr(self.classification_config, key, value)
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration as dictionary."""
        return asdict(self.model_config)
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration as dictionary."""
        return asdict(self.training_config)
    
    def get_classification_config(self) -> Dict[str, Any]:
        """Get classification configuration as dictionary."""
        return asdict(self.classification_config)
    
    def save_all_configs(self):
        """Save all configurations to separate files."""
        self.save_config("model", self.get_model_config())
        self.save_config("training", self.get_training_config())
        self.save_config("classification", self.get_classification_config())
    
    def load_all_configs(self):
        """Load all configurations from files."""
        try:
            model_config = self.load_config("model")
            training_config = self.load_config("training")
            classification_config = self.load_config("classification")
            
            self.model_config = ModelConfig(**model_config)
            self.training_config = TrainingConfig(**training_config)
            self.classification_config = ClassificationConfig(**classification_config)
        except FileNotFoundError:
            # If config files don't exist, use defaults
            pass 