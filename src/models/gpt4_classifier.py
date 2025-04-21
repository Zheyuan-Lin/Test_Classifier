"""
Implements zero-shot classification using GPT-4 for emerging topics.
"""

import os
import re
from typing import List, Dict
import openai
from src.config.constants import CURRICULUM_CATEGORIES

class GPT4ZeroShotClassifier:
    def __init__(self, api_key: str = None):
        """Initialize the GPT-4 classifier."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
        
        openai.api_key = self.api_key
        
    def _create_prompt(self, question: str, categories: List[str]) -> str:
        """Create classification prompt."""
        categories_str = ", ".join(categories)
        return f"""Classify the following medical physics exam question into one of these categories: {categories_str}.

Question: {question}

Format your response as:
Category: [category]
Confidence: [0-100]
Reasoning: [brief explanation]"""

    def classify(self, questions: List[str], categories: List[str] = CURRICULUM_CATEGORIES) -> List[Dict]:
        """Classify a list of questions."""
        results = []
        
        for question in questions:
            try:
                # Make API call
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": self._create_prompt(question, categories)}],
                    temperature=0.3,
                    max_tokens=300
                )
                
                # Parse response
                response_text = response.choices[0].message.content
                category_match = re.search(r"Category:\s*(.*?)(?:\n|$)", response_text)
                confidence_match = re.search(r"Confidence:\s*(\d+)", response_text)
                reasoning_match = re.search(r"Reasoning:\s*(.*)", response_text, re.DOTALL)
                
                result = {
                    "question": question,
                    "predicted_category": category_match.group(1).strip() if category_match else "Unknown",
                    "confidence": int(confidence_match.group(1))/100.0 if confidence_match else 0.0,
                    "reasoning": reasoning_match.group(1).strip() if reasoning_match else ""
                }
                
            except Exception as e:
                result = {
                    "question": question,
                    "predicted_category": "Error",
                    "confidence": 0.0,
                    "reasoning": f"Error: {str(e)}"
                }
                
            results.append(result)
            
        return results
    
    def evaluate(self, questions: List[str], true_labels: List[str]) -> Dict:
        """Evaluate the classifier."""
        results = self.classify(questions)
        predictions = [r["predicted_category"] for r in results]
        
        correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
        accuracy = correct / len(questions)
        
        return {
            "accuracy": accuracy,
            "predictions": list(zip(questions, predictions, true_labels)),
            "results": results
        }
    

##
# Initialize
classifier = GPT4ZeroShotClassifier(api_key="your-api-key")

# Classify single or multiple questions
results = classifier.classify(["What is the half-life of Cobalt-60?"])

# Evaluate
eval_results = classifier.evaluate(test_questions, test_labels)