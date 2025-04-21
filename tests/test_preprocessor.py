"""
Tests for the question preprocessor module.
"""

import pytest
from src.preprocessing.question_preprocessor import ExamQuestionPreprocessor

@pytest.fixture
def preprocessor():
    """Create a preprocessor instance for testing."""
    return ExamQuestionPreprocessor()

def test_extract_stem_mcq(preprocessor):
    """Test stem extraction from MCQ."""
    question = """What is the relationship between dose and survival fraction in the linear-quadratic model?
    A) Linear
    B) Exponential
    C) Quadratic
    D) Logarithmic
    E) None of the above"""
    
    stem, options = preprocessor.extract_stem(question)
    
    assert "What is the relationship between dose and survival fraction in the linear-quadratic model?" in stem
    assert len(options) == 5
    assert "Linear" in options[0]
    assert "Exponential" in options[1]
    assert "Quadratic" in options[2]
    assert "Logarithmic" in options[3]
    assert "None of the above" in options[4]

def test_extract_stem_open_ended(preprocessor):
    """Test stem extraction from open-ended question."""
    question = "Explain the principles of radiation therapy planning."
    
    stem, options = preprocessor.extract_stem(question)
    
    assert stem == question
    assert len(options) == 0

def test_handle_equation(preprocessor):
    """Test equation handling."""
    text = "Calculate the dose using the formula $D = \frac{E}{m}$"
    
    processed_text, equations = preprocessor.handle_equation(text)
    
    assert "[EQUATION_0]" in processed_text
    assert len(equations) == 1
    assert "D = \frac{E}{m}" in equations[0]

def test_preprocess_mcq(preprocessor):
    """Test full preprocessing of MCQ."""
    question_data = {
        "question_text": """What is the relationship between dose and survival fraction in the linear-quadratic model?
        A) Linear
        B) Exponential
        C) Quadratic
        D) Logarithmic
        E) None of the above""",
        "curriculum_category": "Radiation Biology",
        "explanation": "The linear-quadratic model describes cell survival."
    }
    
    processed = preprocessor.preprocess(question_data)
    
    assert processed["question_type"] == "MCQ"
    assert "processed_stem" in processed
    assert "processed_options" in processed
    assert len(processed["processed_options"]) == 5
    assert processed["curriculum_category"] == "Radiation Biology"
    assert processed["explanation"] == "The linear-quadratic model describes cell survival."

def test_preprocess_open_ended(preprocessor):
    """Test full preprocessing of open-ended question."""
    question_data = {
        "question_text": "Explain the principles of radiation therapy planning.",
        "curriculum_category": "Treatment Planning",
        "explanation": "Radiation therapy planning involves multiple steps."
    }
    
    processed = preprocessor.preprocess(question_data)
    
    assert processed["question_type"] == "Open-ended"
    assert "processed_stem" in processed
    assert len(processed["processed_options"]) == 0
    assert processed["curriculum_category"] == "Treatment Planning"
    assert processed["explanation"] == "Radiation therapy planning involves multiple steps." 