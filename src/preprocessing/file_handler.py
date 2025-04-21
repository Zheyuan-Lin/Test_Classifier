import os
import json
from typing import List, Dict
from datetime import datetime

class QuestionFileHandler:
    def __init__(self):
        """Initialize file handler with PDF support."""
        self.supported_formats = ['.txt', '.pdf', '.json', '.csv']
        self.processed_dir = "data/processed"
        
    def read_pdf(self, file_path: str) -> List[Dict]:
        """Read questions from PDF file."""
        import pdfplumber
        
        questions = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                # Split text into individual questions (would need custom logic)
                question_texts = self._split_into_questions(text)
                for q_text in question_texts:
                    questions.append({
                        "question_text": q_text,
                    })
        
        # Save processed questions
        self.save_processed_questions(questions, file_path)
        
        return questions
    
    def save_processed_questions(self, questions: List[Dict], source_file: str):
        """Save processed questions to JSON file."""
        # Create filename based on source file and timestamp
        base_name = os.path.splitext(os.path.basename(source_file))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{base_name}_processed_{timestamp}.json"
        output_path = os.path.join(self.processed_dir, output_file)
        
        # Save to JSON with metadata
        data = {
            "metadata": {
                "source_file": source_file,
                "processing_date": timestamp,
                "num_questions": len(questions)
            },
            "questions": questions
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def load_processed_questions(self, processed_file: str) -> List[Dict]:
        """Load previously processed questions."""
        file_path = os.path.join(self.processed_dir, processed_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data["questions"]
    
    def list_processed_files(self) -> List[str]:
        """List all processed question files."""
        return [f for f in os.listdir(self.processed_dir) if f.endswith('.json')]
    
    def _split_into_questions(self, text: str) -> List[str]:
        """Split text into individual questions."""
        # Implement your question splitting logic here
        # This is a placeholder implementation
        questions = text.split("\n\n")
        return [q.strip() for q in questions if q.strip()]
