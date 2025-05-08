import numpy as np
import pandas as pd
import json
import os
from typing import List, Dict, Tuple, Optional

# SKLearn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.class_weight import compute_class_weight

# NLP preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import string

# Transformers import
from sentence_transformers import SentenceTransformer

# Persistence
from joblib import dump, load

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')


class TextPreprocessor:
    """Simple text preprocessing for physics exam questions."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text."""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        tokens = [word for word in tokens if word not in self.stop_words and len(word) >= 2]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
            
        # Join tokens back into a string
        return ' '.join(tokens)
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """Process a list of texts."""
        return [self.preprocess_text(text) for text in texts]


class PhysicsExamClassifier:
    """
    Simplified classifier for physics exam questions using transformer embeddings
    and ensemble modeling with SVM, RandomForest, and LogisticRegression.
    """
    
    def __init__(self, 
                 model_name: str = "all-mpnet-base-v2",
                 random_state: int = 42,
                 output_dir: str = "./models"):
        
        # Basic parameters
        self.random_state = random_state
        self.output_dir = output_dir
        
        # Text preprocessor
        self.preprocessor = TextPreprocessor()
        
        # Initialize encoders
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # Enhanced TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2,
            max_df=0.85,
            sublinear_tf=True,
            use_idf=True,
            smooth_idf=True
        )
        
        # Enhanced dimensionality reduction
        self.svd = TruncatedSVD(n_components=100, random_state=random_state)
        
        # Enhanced transformer model with better configuration
        self.sentence_transformer = SentenceTransformer(model_name)
        self.sentence_transformer.max_seq_length = 256  # Increased from default
        self.sentence_transformer.batch_size = 16  # Reduced for better stability
        
        # Enhanced individual models with better transformer-specific configurations
        self.base_models = {
            'svm_tfidf': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                cache_size=2000,
                random_state=random_state
            ),
            'svm_transformer': SVC(
                kernel='rbf',
                C=2.0,  # Increased for better margin
                gamma='scale',
                class_weight='balanced',
                probability=True,
                cache_size=2000,
                random_state=random_state
            ),
            'rf_tfidf': RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                random_state=random_state,
                n_jobs=-1,
                bootstrap=True
            ),
            'rf_transformer': RandomForestClassifier(
                n_estimators=500,  # Increased for better feature utilization
                max_depth=20,  # Increased for transformer features
                min_samples_split=3,  # Reduced for better feature utilization
                min_samples_leaf=1,  # Reduced for better feature utilization
                max_features='sqrt',
                class_weight='balanced',
                random_state=random_state,
                n_jobs=-1,
                bootstrap=True
            ),
            'lr_tfidf': LogisticRegression(
                C=1.0,
                penalty='elasticnet',
                solver='saga',
                max_iter=2000,
                class_weight='balanced',
                random_state=random_state,
                n_jobs=-1,
                l1_ratio=0.5
            ),
            'lr_transformer': LogisticRegression(
                C=2.0,  # Increased for better margin
                penalty='elasticnet',
                solver='saga',
                max_iter=3000,  # Increased for better convergence
                class_weight='balanced',
                random_state=random_state,
                n_jobs=-1,
                l1_ratio=0.3  # Adjusted for better regularization
            )
        }
        
        # Ensemble model
        self.ensemble_model = None
        
        # Storage for features
        self.feature_vectors = {}
        
    def load_data(self, json_path: str, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and split data from JSON file."""
        print("Loading data...")
        
        # Load JSON data
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(data['dataset'])
        
        # Count examples per category
        category_counts = df['category'].value_counts()
        print("Category distribution:")
        print(category_counts)
        
        # Filter out categories with very few examples
        min_samples = 3
        valid_categories = category_counts[category_counts >= min_samples].index
        df = df[df['category'].isin(valid_categories)]
        
        # Split data
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=df['category']
        )
        
        print(f"Training set size: {len(train_df)}")
        print(f"Test set size: {len(test_df)}")
        
        # Fit label encoder
        self.label_encoder.fit(df['category'].values)
        
        return train_df, test_df
    
    def create_tfidf_features(self, texts: List[str], fit: bool = False) -> np.ndarray:
        """Create TF-IDF features with dimensionality reduction."""
        print("Creating TF-IDF features...")
        
        # Preprocess texts
        processed_texts = self.preprocessor.preprocess_texts(texts)
        
        if fit:
            # Fit and transform
            tfidf_features = self.tfidf_vectorizer.fit_transform(processed_texts)
            tfidf_features = self.svd.fit_transform(tfidf_features)
            tfidf_features = self.scaler.fit_transform(tfidf_features)
        else:
            # Transform with fitted components
            tfidf_features = self.tfidf_vectorizer.transform(processed_texts)
            tfidf_features = self.svd.transform(tfidf_features)
            tfidf_features = self.scaler.transform(tfidf_features)
        
        # Ensure non-negative values
        tfidf_features = np.maximum(tfidf_features, 0)
        
        return tfidf_features
    
    def create_transformer_features(self, texts: List[str]) -> np.ndarray:
        """Create enhanced transformer embeddings."""
        print("Creating transformer embeddings...")
        
        # Preprocess texts
        processed_texts = self.preprocessor.preprocess_texts(texts)
        
        # Get embeddings with enhanced configuration
        embeddings = self.sentence_transformer.encode(
            processed_texts,
            show_progress_bar=True,
            batch_size=self.sentence_transformer.batch_size,
            normalize_embeddings=True,  # Added normalization
            convert_to_numpy=True
        )
        
        # Apply dimensionality reduction to transformer features
        if not hasattr(self, 'transformer_svd'):
            self.transformer_svd = TruncatedSVD(n_components=128, random_state=self.random_state)
            embeddings = self.transformer_svd.fit_transform(embeddings)
        else:
            embeddings = self.transformer_svd.transform(embeddings)
        
        # Scale features
        if not hasattr(self, 'transformer_scaler'):
            self.transformer_scaler = StandardScaler()
            embeddings = self.transformer_scaler.fit_transform(embeddings)
        else:
            embeddings = self.transformer_scaler.transform(embeddings)
        
        return embeddings
    
    def prepare_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
        """Prepare features for training and testing."""
        print("Preparing features...")
        
        # Create features
        self.feature_vectors = {
            'tfidf': {
                'train': self.create_tfidf_features(train_df['question_text'].tolist(), fit=True),
                'test': self.create_tfidf_features(test_df['question_text'].tolist())
            },
            'transformer': {
                'train': self.create_transformer_features(train_df['question_text'].tolist()),
                'test': self.create_transformer_features(test_df['question_text'].tolist())
            }
        }
        
        return self.feature_vectors
    
    def train_models(self, features: Dict, train_df: pd.DataFrame) -> Dict:
        """Train all models on the prepared features."""
        print("Training models...")
        
        # Encode labels
        y = self.label_encoder.transform(train_df['category'].values)
        
        # Calculate class weights with balanced strategy
        class_weights = compute_class_weight(
            'balanced',  # Changed from balanced_subsample to balanced
            classes=np.unique(y),
            y=y
        )
        class_weight_dict = dict(zip(np.unique(y), class_weights))
        
        # Results dictionary
        results = {}
        trained_models = {}
        
        # Train individual models
        for model_name, model in self.base_models.items():
            print(f"\nTraining {model_name}...")
            
            # Get feature type from model name
            feature_type = model_name.split('_')[1]
            
            # Get features and train
            X = features[feature_type]['train']
            
            # Update class weights if model supports it
            if hasattr(model, 'class_weight'):
                model.class_weight = class_weight_dict
            
            # Add feature-specific training parameters
            if feature_type == 'transformer':
                if isinstance(model, SVC):
                    model.C = 2.0  # Increased for transformer features
                elif isinstance(model, RandomForestClassifier):
                    model.n_estimators = 500  # Increased for transformer features
            
            model.fit(X, y)
            
            # Evaluate on training set
            y_pred = model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            
            results[model_name] = accuracy
            trained_models[model_name] = model
            
            print(f"{model_name} training accuracy: {accuracy:.3f}")
        
        # Create and train ensemble model with enhanced weighted voting
        print("\nTraining ensemble model...")
        
        # Define voting classifier with enhanced weights
        estimators = [(name, model) for name, model in trained_models.items()]
        weights = [results[name] ** 2 for name, _ in estimators]  # Square weights for stronger emphasis
        
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=weights,
            n_jobs=-1
        )
        
        # Create combined features with enhanced weighting
        combined_features = np.hstack([
            features['tfidf']['train'] * 0.4,  # Reduced TF-IDF weight
            features['transformer']['train'] * 0.6  # Increased transformer weight
        ])
        
        # Train ensemble
        self.ensemble_model.fit(combined_features, y)
        
        # Evaluate ensemble
        y_pred_ensemble = self.ensemble_model.predict(combined_features)
        ensemble_accuracy = accuracy_score(y, y_pred_ensemble)
        
        results['ensemble'] = ensemble_accuracy
        print(f"Ensemble model training accuracy: {ensemble_accuracy:.3f}")
        
        return results
    
    def evaluate_models(self, features: Dict, test_df: pd.DataFrame) -> Dict:
        """Evaluate all models on test data."""
        print("\nEvaluating models on test data...")
        print("=" * 80)
        
        # Encode labels
        y = self.label_encoder.transform(test_df['category'].values)
        
        # Results dictionary
        results = {}
        
        # Group models by feature type
        feature_types = ['tfidf', 'transformer']
        for feature_type in feature_types:
            print(f"\nFeature Type: {feature_type.upper()}")
            print("-" * 40)
            
            # Evaluate individual models for this feature type
            for model_name, model in self.base_models.items():
                if model_name.endswith(feature_type):
                    print(f"\nModel: {model_name}")
                    print("-" * 20)
                    
                    # Get features and predict
                    X = features[feature_type]['test']
                    y_pred = model.predict(X)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y, y_pred)
                    report = classification_report(y, y_pred, target_names=self.label_encoder.classes_, output_dict=True)
                    
                    results[model_name] = {
                        'accuracy': accuracy,
                        'report': report
                    }
                    
                    # Print detailed metrics
                    print(f"Accuracy: {accuracy:.3f}")
                    print("\nPer-class metrics:")
                    for class_name in self.label_encoder.classes_:
                        metrics = report[class_name]
                        print(f"\n{class_name}:")
                        print(f"  Precision: {metrics['precision']:.3f}")
                        print(f"  Recall: {metrics['recall']:.3f}")
                        print(f"  F1-score: {metrics['f1-score']:.3f}")
                        print(f"  Support: {metrics['support']}")
        
        # Evaluate ensemble model
        if self.ensemble_model:
            print("\nEnsemble Model Evaluation")
            print("=" * 40)
            
            # Create combined features
            combined_features = np.hstack([
                features['tfidf']['test'],
                features['transformer']['test']
            ])
            
            # Predict
            y_pred = self.ensemble_model.predict(combined_features)
            
            # Calculate metrics
            accuracy = accuracy_score(y, y_pred)
            report = classification_report(y, y_pred, target_names=self.label_encoder.classes_, output_dict=True)
            
            results['ensemble'] = {
                'accuracy': accuracy,
                'report': report
            }
            
            # Print detailed metrics
            print(f"Ensemble Model Accuracy: {accuracy:.3f}")
            print("\nPer-class metrics:")
            for class_name in self.label_encoder.classes_:
                metrics = report[class_name]
                print(f"\n{class_name}:")
                print(f"  Precision: {metrics['precision']:.3f}")
                print(f"  Recall: {metrics['recall']:.3f}")
                print(f"  F1-score: {metrics['f1-score']:.3f}")
                print(f"  Support: {metrics['support']}")
        
        # Print summary
        print("\nModel Performance Summary")
        print("=" * 40)
        for model_name, result in results.items():
            print(f"{model_name}: {result['accuracy']:.3f}")
        
        return results
    
    def predict(self, text: str, use_ensemble: bool = True) -> Dict:
        """Predict category for a single physics exam question."""
        # Extract features
        tfidf_features = self.create_tfidf_features([text])
        transformer_features = self.create_transformer_features([text])
        
        if use_ensemble and self.ensemble_model:
            # Combine features
            combined_features = np.hstack([tfidf_features, transformer_features])
            
            # Predict
            label_idx = self.ensemble_model.predict(combined_features)[0]
            probas = self.ensemble_model.predict_proba(combined_features)[0]
            
            # Get confidence scores
            confidence_scores = {}
            for idx, prob in enumerate(probas):
                category = self.label_encoder.inverse_transform([idx])[0]
                confidence_scores[category] = prob
            
            # Sort by probability
            confidence_scores = dict(sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True))
            
            return {
                'predicted_category': self.label_encoder.inverse_transform([label_idx])[0],
                'confidence_scores': confidence_scores
            }
        else:
            # Use SVM with transformer features (usually best single model)
            model = self.base_models['svm_transformer']
            
            # Predict
            label_idx = model.predict(transformer_features)[0]
            probas = model.predict_proba(transformer_features)[0]
            
            # Get confidence scores
            confidence_scores = {}
            for idx, prob in enumerate(probas):
                category = self.label_encoder.inverse_transform([idx])[0]
                confidence_scores[category] = prob
            
            # Sort by probability
            confidence_scores = dict(sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True))
            
            return {
                'predicted_category': self.label_encoder.inverse_transform([label_idx])[0],
                'confidence_scores': confidence_scores
            }
    
    def save_model(self, output_dir: Optional[str] = None):
        """Save all model components."""
        save_dir = output_dir or self.output_dir
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Saving model to {save_dir}...")
        
        # Save preprocessor and feature extractors
        dump(self.preprocessor, os.path.join(save_dir, 'preprocessor.joblib'))
        dump(self.tfidf_vectorizer, os.path.join(save_dir, 'tfidf_vectorizer.joblib'))
        dump(self.svd, os.path.join(save_dir, 'svd.joblib'))
        dump(self.scaler, os.path.join(save_dir, 'scaler.joblib'))
        dump(self.label_encoder, os.path.join(save_dir, 'label_encoder.joblib'))
        
        # Save individual models
        for name, model in self.base_models.items():
            dump(model, os.path.join(save_dir, f'{name}.joblib'))
        
        # Save ensemble model
        if self.ensemble_model:
            dump(self.ensemble_model, os.path.join(save_dir, 'ensemble_model.joblib'))
            
        # Save model info
        model_info = {
            'classes': self.label_encoder.classes_.tolist(),
            'sentence_transformer': self.sentence_transformer.get_config_dict()['model_name'] if hasattr(self.sentence_transformer, 'get_config_dict') else 'all-mpnet-base-v2',
            'version': '1.0'
        }
        
        with open(os.path.join(save_dir, 'model_info.json'), 'w') as f:
            json.dump(model_info, f)
            
        print("Model saved successfully")
    
    def load_model(self, model_dir: str):
        """Load all model components."""
        print(f"Loading model from {model_dir}...")
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Load preprocessor and feature extractors
        self.preprocessor = load(os.path.join(model_dir, 'preprocessor.joblib'))
        self.tfidf_vectorizer = load(os.path.join(model_dir, 'tfidf_vectorizer.joblib'))
        self.svd = load(os.path.join(model_dir, 'svd.joblib'))
        self.scaler = load(os.path.join(model_dir, 'scaler.joblib'))
        self.label_encoder = load(os.path.join(model_dir, 'label_encoder.joblib'))
        
        # Load model info
        with open(os.path.join(model_dir, 'model_info.json'), 'r') as f:
            model_info = json.load(f)
            transformer_name = model_info.get('sentence_transformer', 'all-mpnet-base-v2')
        
        # Load sentence transformer
        try:
            self.sentence_transformer = SentenceTransformer(transformer_name)
        except Exception as e:
            print(f"Warning: Could not load transformer model {transformer_name}. Using default model.")
            self.sentence_transformer = SentenceTransformer('all-mpnet-base-v2')
        
        # Load individual models
        for name in self.base_models.keys():
            model_path = os.path.join(model_dir, f'{name}.joblib')
            if os.path.exists(model_path):
                self.base_models[name] = load(model_path)
        
        # Load ensemble model
        ensemble_path = os.path.join(model_dir, 'ensemble_model.joblib')
        if os.path.exists(ensemble_path):
            self.ensemble_model = load(ensemble_path)
            
        print("Model loaded successfully")
    
    def run_pipeline(self, json_path: str, test_size: float = 0.2, output_dir: Optional[str] = None) -> Dict:
        """Run the complete training and evaluation pipeline."""
        start_time = pd.Timestamp.now()
        print(f"Starting pipeline at {start_time}")
        
        # Set output directory
        save_dir = output_dir or self.output_dir
        
        # Load and split data
        train_df, test_df = self.load_data(json_path, test_size)
        
        # Prepare features
        features = self.prepare_features(train_df, test_df)
        
        # Train models
        train_results = self.train_models(features, train_df)
        
        # Evaluate models
        test_results = self.evaluate_models(features, test_df)
        
        # Save model
        self.save_model(save_dir)
        
        # Calculate elapsed time
        end_time = pd.Timestamp.now()
        elapsed = end_time - start_time
        print(f"Pipeline completed in {elapsed}")
        
        return {
            'train_results': train_results,
            'test_results': test_results,
            'elapsed_time': str(elapsed)
        }


# Example usage
if __name__ == "__main__":
    # Initialize classifier
    classifier = PhysicsExamClassifier(
        model_name="all-mpnet-base-v2",
        random_state=42,
        output_dir="./traditional_model"
    )
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Run pipeline
    results = classifier.run_pipeline(
        json_path = os.path.join(current_dir, '..', '..', 'data', 'processed', 'synthetic.json'),
        test_size=0.2
    )
    
    # Print best model
    if 'test_results' in results:
        best_model = max(results['test_results'].items(), 
                          key=lambda x: x[1]['accuracy'] if isinstance(x[1], dict) else x[1])
        print(f"\nBest model: {best_model[0]}")
        