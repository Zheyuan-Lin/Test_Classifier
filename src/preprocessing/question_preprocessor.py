"""
Handles preprocessing of exam questions, focusing on:
1. Stem extraction (separating stem from options in MCQs)
2. Equation handling (replacing LaTeX with placeholders)
3. Standardizing format across question types
4. Preparing data for classification
"""

import re
from typing import List, Dict, Tuple

class ExamQuestionPreprocessor:
    def __init__(self, handle_equations: bool = True):
        self.handle_equations = handle_equations
        # Regex patterns for different question formats
        self.mcq_pattern = r"(.*?)\s*(?:A\)|A\.)(.*?)(?:B\)|B\.)(.*?)(?:C\)|C\.)(.*?)(?:D\)|D\.)(.*?)(?:E\)|E\.)?(.*?)$"
        self.latex_pattern = r"\$\$(.*?)\$\$|\$(.*?)\$"
    
    def extract_stem(self, question_text: str) -> Tuple[str, List[str]]:
        """Extract the question stem and options from MCQ."""
        # Check if it's an MCQ
        if re.search(r"(?:A\)|A\.)|(?:B\)|B\.)|(?:C\)|C\.)|(?:D\)|D\.)", question_text):
            match = re.search(self.mcq_pattern, question_text, re.DOTALL)
            if match:
                stem = match.group(1).strip()
                options = [match.group(i).strip() for i in range(2, 7) if match.group(i)]
                return stem, options
        
        # If not recognized as MCQ, return the whole as stem
        return question_text.strip(), []
    
    def handle_equation(self, text: str) -> Tuple[str, List[str]]:
        """Replace LaTeX equations with placeholders and return mapping."""
        if not self.handle_equations:
            return text, []
        
        equation_map = []
        
        def replace_equation(match):
            eq = match.group(1) or match.group(2)
            placeholder = f"[EQUATION_{len(equation_map)}]"
            equation_map.append(eq)
            return placeholder
        
        processed_text = re.sub(self.latex_pattern, replace_equation, text)
        return processed_text, equation_map
    
    def preprocess(self, question_data: Dict) -> Dict:
        """Main preprocessing method."""
        question_text = question_data.get("question_text", "")
        
        # Extract stem and options
        stem, options = self.extract_stem(question_text)
        
        # Handle equations
        processed_stem, stem_equations = self.handle_equation(stem)
        processed_options = []
        option_equations = []
        
        for option in options:
            proc_option, equations = self.handle_equation(option)
            processed_options.append(proc_option)
            option_equations.extend(equations)
        
        # Prepare the processed question data
        processed_data = {
            "original_text": question_text,
            "processed_stem": processed_stem,
            "processed_options": processed_options,
            "stem_equations": stem_equations,
            "option_equations": option_equations,
            "question_type": "MCQ" if options else "Open-ended",
            "curriculum_category": question_data.get("curriculum_category", ""),
            "explanation": question_data.get("explanation", "")
        }
        
        return processed_data

    def prepare_for_classification(self, processed_questions: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        Convert preprocessed questions into format needed for BaselineClassifier.
        
        Args:
            processed_questions: List of preprocessed question dictionaries
            
        Returns:
            texts: List of processed text ready for classification
            labels: List of curriculum categories
        """
        texts = []
        labels = []
        
        for question in processed_questions:
            # Combine stem and options into a single text
            text_parts = [question["processed_stem"]]
            
            # Add options if they exist (for MCQs)
            if question["processed_options"]:
                # Add options with their labels
                option_labels = ["A", "B", "C", "D", "E"]
                for i, option in enumerate(question["processed_options"]):
                    if i < len(option_labels):
                        text_parts.append(f"{option_labels[i]}) {option}")
            
            # Join all parts with spaces
            combined_text = " ".join(text_parts)
            
            # Add to lists if curriculum category exists
            if question["curriculum_category"]:
                texts.append(combined_text)
                labels.append(question["curriculum_category"])
        
        return texts, labels

    def batch_process_for_classification(self, raw_questions: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        Process a batch of raw questions and prepare them for classification.
        
        Args:
            raw_questions: List of raw question dictionaries
            
        Returns:
            texts: List of processed text ready for classification
            labels: List of curriculum categories
        """
        # First preprocess all questions
        processed_questions = [self.preprocess(q) for q in raw_questions]
        
        # Then prepare for classification
        return self.prepare_for_classification(processed_questions) 