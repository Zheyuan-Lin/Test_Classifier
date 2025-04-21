# AI-Powered Exam Question Classification Pipeline

This project implements a comprehensive pipeline for classifying medical physics exam questions according to AAPM/ASTRO curriculum categories. The system combines multiple classification approaches:

1. Traditional methods (TF-IDF + SVM/Naive Bayes)
2. Transformer-based models (SciBERT/BERT)
3. GPT-4 zero-shot classification

## Features

- Preprocessing of exam questions with special handling for:
  - Multiple-choice questions (stem extraction)
  - LaTeX equations
  - Question explanations
- Multiple classification models with confidence-based fallback
- Comprehensive evaluation metrics:
  - Distractor robustness
  - Cross-format accuracy
  - Novel term adaptation
- Model explanation using LIME
- Curriculum-aware training and evaluation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/exam-classifier.git
cd exam-classifier
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up OpenAI API key:
```bash
export OPENAI_API_KEY=your_api_key_here
```

## Project Structure

```
src/
├── config/
│   └── constants.py         # Configuration constants
├── preprocessing/
│   └── question_preprocessor.py  # Question preprocessing
├── models/
│   ├── baseline_classifier.py    # Traditional classification
│   ├── transformer_classifier.py # Transformer-based classification
│   ├── gpt4_classifier.py        # GPT-4 zero-shot classification
│   └── exam_classifier.py        # Main classifier class
└── utils/                  # Utility functions (to be added)
```

## Usage

### Basic Classification

```python
from src.models.exam_classifier import ExamQuestionClassifier

# Initialize classifier
classifier = ExamQuestionClassifier()

# Classify a single question
result = classifier.classify(
    "What is the relationship between dose and survival fraction in the linear-quadratic model?"
)

# Classify multiple questions
results = classifier.batch_classify([
    "Question 1...",
    "Question 2..."
])
```

### Training Models

```python
import pandas as pd

# Load your question dataset
questions_df = pd.read_csv("questions.csv")

# Train baseline model
baseline_results, (X_train, X_test, y_train, y_test) = classifier.train_baseline(questions_df)

# Train transformer model
transformer_results, _ = classifier.train_transformer(questions_df)
```

### Evaluation

```python
# Evaluate distractor robustness
distractor_results = classifier.evaluate_distractor_robustness(questions_df)

# Evaluate cross-format accuracy
format_results = classifier.evaluate_cross_format_accuracy(questions_df)

# Evaluate novel term adaptation
novel_results = classifier.evaluate_novel_term_adaptation(
    questions_df,
    novel_terms=["proton therapy", "FLASH"]
)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. # Test_Classifier
