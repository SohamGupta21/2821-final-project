"""Base classes for model adapters and observation languages."""

from abc import ABC, abstractmethod
from typing import Any, List, Dict, Callable, Optional, Set
from dataclasses import dataclass, field
import numpy as np

from ..core.atoms import Atom, Constant, Variable


class ModelAdapter(ABC):
    """
    Abstract adapter interface for any ML model.
    
    Implement this to support a new model type. The adapter provides
    a unified interface for getting predictions from any model.
    
    Example:
        >>> class MyCustomAdapter(ModelAdapter):
        ...     def predict(self, instance):
        ...         return self.model.my_predict_method(instance)
        ...     def predict_proba(self, instance):
        ...         return self.model.my_proba_method(instance)
        ...     @property
        ...     def model_type(self):
        ...         return "custom"
    """
    
    @abstractmethod
    def predict(self, instance: Any) -> Any:
        """
        Get model prediction for an instance.
        
        Args:
            instance: Input data in model's expected format
            
        Returns:
            Model prediction (class label, probability, etc.)
        """
        pass
    
    @abstractmethod
    def predict_proba(self, instance: Any) -> Optional[np.ndarray]:
        """
        Get prediction probabilities if available.
        
        Returns:
            Array of probabilities or None if not supported
        """
        pass
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return the model type identifier."""
        pass


class ObservationLanguage(ABC):
    """
    Abstract observation language for converting data to logical atoms.
    
    Defines the vocabulary (predicates) and how to generate facts
    from model inputs/outputs. This is the key abstraction that allows
    the system to work with any data type.
    
    The observation language determines:
    1. What predicates are available (e.g., predict, feature, has_word)
    2. How to generate facts from raw data
    3. How to check if a given atom is true
    
    Example:
        >>> class MyLanguage(ObservationLanguage):
        ...     def get_predicates(self):
        ...         return ["predict", "my_predicate"]
        ...     def generate_instance_facts(self, ...):
        ...         # Convert instance to atoms
        ...     def query_atom(self, atom, instances, predictions):
        ...         # Check if atom is true
    """
    
    @abstractmethod
    def get_predicates(self) -> List[str]:
        """Return list of predicates in this observation language."""
        pass
    
    @abstractmethod
    def generate_instance_facts(
        self,
        instance_id: Any,
        instance_data: Any,
        prediction: Any,
        label_name: str
    ) -> List[Atom]:
        """
        Generate all observation facts for an instance.
        
        Args:
            instance_id: Unique identifier for the instance
            instance_data: The instance's data representation
            prediction: Model's prediction for this instance
            label_name: Human-readable label name
            
        Returns:
            List of ground atoms representing observations
        """
        pass
    
    @abstractmethod
    def query_atom(self, atom: Atom, instances: Dict[Any, Any], predictions: Dict[Any, Any]) -> bool:
        """
        Check if an atom is true given the data.
        
        Args:
            atom: Ground atom to check
            instances: Instance data store
            predictions: Prediction cache
            
        Returns:
            True if atom is true, False otherwise
        """
        pass
    
    @property
    @abstractmethod
    def language_name(self) -> str:
        """Return the language identifier."""
        pass


@dataclass
class PredicateSpec:
    """
    Specification for a predicate in the observation language.
    
    Provides metadata about predicates for documentation and validation.
    """
    name: str
    arity: int
    description: str
    argument_names: List[str]
    is_target: bool = False  # True for prediction predicates
    
    def __str__(self) -> str:
        args = ", ".join(self.argument_names)
        return f"{self.name}({args})"
    
    def validate_atom(self, atom: Atom) -> bool:
        """Check if an atom matches this predicate specification."""
        return (
            atom.predicate == self.name and
            len(atom.arguments) == self.arity
        )

