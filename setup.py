"""
Setup configuration for the exam classification package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="exam-classifier",
    version="0.1.0",
    author="Zheyuan Lin",
    author_email="zlin252@emory.edu",
    description="AI-powered exam question classification pipeline for medical physics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zheyuan-lin/exam-classifier",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.1",
            "pytest-mock>=3.6.1",
            "black>=21.7b0",
            "isort>=5.9.3",
            "flake8>=3.9.2",
            "mypy>=0.910",
        ],
    },
    entry_points={
        "console_scripts": [
            "exam-classifier=src.cli:main",
        ],
    },
) 