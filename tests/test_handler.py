"""
Main entry point for the exam question classification system.
"""

from src.preprocessing.file_handler import QuestionFileHandler
from src.utils.logger import get_logger

logger = get_logger(__name__)

def process_pdf_file(pdf_path: str):
    logger.info(f"Processing PDF file: {pdf_path}")
    
    # Read questions from PDF
    file_handler = QuestionFileHandler()
    questions = file_handler.read_pdf(pdf_path)
    logger.info(f"Extracted {len(questions)} questions from PDF")

def main():
    """Main function."""
    # Example usage
    results = process_pdf_file(
        pdf_path="data/raw/exam_questions.pdf",
    )

if __name__ == "__main__":
    main() 