from .base import ModelAdapter, ObservationLanguage, PredicateSpec
from .adapters import (
    SklearnAdapter,
    PyTorchAdapter,
    TensorFlowAdapter,
    GenericAdapter,
    detect_and_create_adapter
)
from .languages import (
    TabularObservationLanguage,
    ImageObservationLanguage,
    TextObservationLanguage
)
from .universal_oracle import UniversalOracle
from .factory import create_explainer, explain_model, ExplainabilityResult

__all__ = [
    # Base classes
    "ModelAdapter",
    "ObservationLanguage",
    "PredicateSpec",
    # Adapters
    "SklearnAdapter",
    "PyTorchAdapter",
    "TensorFlowAdapter",
    "GenericAdapter",
    "detect_and_create_adapter",
    # Languages
    "TabularObservationLanguage",
    "ImageObservationLanguage",
    "TextObservationLanguage",
    # Oracle
    "UniversalOracle",
    # Factory (main API)
    "create_explainer",
    "explain_model",
    "ExplainabilityResult",
]


