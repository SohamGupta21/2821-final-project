"""Base classes for model adapters and observation languages."""

from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional
from dataclasses import dataclass
import numpy as np

from ..core.atoms import Atom


class ModelAdapter(ABC):

    
    @abstractmethod
    def predict(self, instance: Any) -> Any:

        pass
    
    @abstractmethod
    def predict_proba(self, instance: Any) -> Optional[np.ndarray]:

        pass
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        pass


class ObservationLanguage(ABC):
    
    @abstractmethod
    def get_predicates(self) -> List[str]:
        pass
    
    @abstractmethod
    def generate_instance_facts(
        self,
        instance_id: Any,
        instance_data: Any,
        prediction: Any,
        label_name: str
    ) -> List[Atom]:
        pass
    
    @abstractmethod
    def query_atom(self, atom: Atom, instances: Dict[Any, Any], predictions: Dict[Any, Any]) -> bool:

        pass
    
    @property
    @abstractmethod
    def language_name(self) -> str:
        pass


@dataclass
class PredicateSpec:
    name: str
    arity: int
    description: str
    argument_names: List[str]
    is_target: bool = False  # True for prediction predicates
    
    def __str__(self) -> str:
        args = ", ".join(self.argument_names)
        return f"{self.name}({args})"
    
    def validate_atom(self, atom: Atom) -> bool:
        return (
            atom.predicate == self.name and
            len(atom.arguments) == self.arity
        )


