import pdfplumber
from typing import List, Optional
import json

class PDFQuestionExtractor:
    """
    Extracts question texts from a PDF file and can save them in a tokenizable format.
    """

    def __init__(self, question_splitter=None):
        """
        Args:
            question_splitter: Optional function to split text into questions.
                               If None, uses double newlines as separator.
        """
        self.question_splitter = question_splitter or self.default_splitter

    def extract_questions(self, pdf_path: str) -> List[str]:
        """
        Extracts questions from a PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of question strings.
        """
        questions = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    questions.extend(self.question_splitter(text))
        return [q.strip() for q in questions if q.strip()]

    @staticmethod
    def default_splitter(text: str) -> List[str]:
        """
        Default method to split text into questions.
        Splits on double newlines.
        """
        return text.split('\n\n')

    def save_questions(
        self, questions: List[str], out_path: str, fmt: str = "txt"
    ) -> None:
        """
        Save questions to a file in a tokenizable format.

        Args:
            questions: List of question strings.
            out_path: Output file path.
            fmt: Format to save ('txt' or 'json').
        """
        if fmt == "txt":
            # Save one question per line (or use '\n\n'.join for double newline)
            with open(out_path, "w", encoding="utf-8") as f:
                for q in questions:
                    f.write(q.strip() + "\n")
        elif fmt == "json":
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"questions": questions}, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError("Unsupported format. Use 'txt' or 'json'.")

# Example usage:
# extractor = PDFQuestionExtractor()
# questions = extractor.extract_questions("path/to/questions.pdf")
# extractor.save_questions(questions, "questions.txt", fmt="txt")
# extractor.save_questions(questions, "questions.json", fmt="json") 