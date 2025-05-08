import re
import nltk
from typing import List, Dict
import json

class TextPreprocessor:
    def __init__(self):
        # Download required NLTK data
        required_data = [
            'punkt',           # For word tokenization
            'wordnet',         # For lemmatization
            'averaged_perceptron_tagger',  # For better lemmatization
            'punkt_tab'        # Required by punkt tokenizer
        ]
        
        for data in required_data:
            try:
                # Try to find the data
                nltk.data.find(f'tokenizers/{data}')
            except LookupError:
                # If not found, download it
                print(f"Downloading NLTK data: {data}")
                nltk.download(data, quiet=False)  # Set quiet=False to see download progress
        
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.question_pattern = re.compile(r'^t\d+\.', re.IGNORECASE)

    def clean(self, text: str) -> str:
        """Clean a single text string."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return ' '.join(tokens)

    def split_into_questions(self, text: str) -> List[str]:
        """Split text into individual questions."""
        lines = text.split('\n')
        questions = []
        current_question = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts with a question number (e.g., T1., T2., etc.)
            if self.question_pattern.match(line):
                if current_question:
                    questions.append(' '.join(current_question))
                    current_question = []
                current_question.append(line)
            else:
                current_question.append(line)
                
        if current_question:
            questions.append(' '.join(current_question))
            
        return questions

    def process_questions(self, text: str) -> Dict[str, List[str]]:
        """Process text and return cleaned questions."""
        # Split into individual questions
        raw_questions = self.split_into_questions(text)
        
        # Clean each question
        cleaned_questions = [self.clean(q) for q in raw_questions]
        
        return {
            "questions": cleaned_questions
        }

    def save_questions(self, questions: Dict[str, List[str]], output_path: str):
        """Save processed questions to a JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)

    def batch_clean(self, texts: List[str]) -> List[str]:
        return [self.clean(t) for t in texts]