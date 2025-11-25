"""Setup script for Shapiro Model Inference."""

from setuptools import setup, find_packages

setup(
    name="shapiro-model-inference",
    version="0.1.0",
    description="Shapiro's Model Inference Algorithm for Neural Network Explainability",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
    ],
    python_requires=">=3.7",
)

