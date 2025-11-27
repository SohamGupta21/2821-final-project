"""
Shapiro's Model Inference for Neural Network Explainability.

A plug-and-play explainability system that learns interpretable rules
from any machine learning model.

Quick Start:
    >>> from src import create_explainer, explain_model
    >>> 
    >>> # One-liner explanation
    >>> print(explain_model(model, X))
    >>> 
    >>> # Detailed analysis
    >>> result = create_explainer(model, X, feature_names=["age", "income"])
    >>> print(result.summary())
    >>> for rule in result.rules:
    ...     print(rule)

Supported Models:
    - scikit-learn (RandomForest, SVM, etc.)
    - PyTorch (nn.Module)
    - TensorFlow/Keras
    - Any model with a predict() method

Data Types:
    - Tabular/structured data
    - Images
    - Text
"""

__version__ = "0.1.0"

# Core logical reasoning
from .core import (
    Atom,
    Term,
    Variable,
    Constant,
    Clause,
    Theory,
    ResolutionEngine,
    ProofNode,
)

# Inference engine
from .inference import (
    Oracle,
    NNOracle,
    ModelInference,
    ContradictionBacktracer,
    RefinementOperator,
)

# Plug-and-play API (main interface)
from .adapters import (
    # Main functions
    create_explainer,
    explain_model,
    ExplainabilityResult,
    # Adapters
    ModelAdapter,
    SklearnAdapter,
    PyTorchAdapter,
    TensorFlowAdapter,
    GenericAdapter,
    detect_and_create_adapter,
    # Observation languages
    ObservationLanguage,
    TabularObservationLanguage,
    ImageObservationLanguage,
    TextObservationLanguage,
    # Oracle
    UniversalOracle,
)

__all__ = [
    # Version
    "__version__",
    # Main API (plug-and-play)
    "create_explainer",
    "explain_model",
    "ExplainabilityResult",
    # Core
    "Atom",
    "Term",
    "Variable",
    "Constant",
    "Clause",
    "Theory",
    "ResolutionEngine",
    "ProofNode",
    # Inference
    "Oracle",
    "NNOracle",
    "ModelInference",
    "ContradictionBacktracer",
    "RefinementOperator",
    # Adapters
    "ModelAdapter",
    "SklearnAdapter",
    "PyTorchAdapter",
    "TensorFlowAdapter",
    "GenericAdapter",
    "detect_and_create_adapter",
    # Observation languages
    "ObservationLanguage",
    "TabularObservationLanguage",
    "ImageObservationLanguage",
    "TextObservationLanguage",
    # Universal oracle
    "UniversalOracle",
]
