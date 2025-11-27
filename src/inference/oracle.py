"""Oracle interface for querying black-box models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import numpy as np
from ..core.atoms import Atom, Constant, Variable


class Oracle(ABC):
    """Abstract base class for oracles that can answer queries about ground atoms."""
    
    @abstractmethod
    def query(self, atom: Atom) -> bool:
        """
        Query the oracle about a ground atom.
        
        Args:
            atom: A ground atom (no variables)
        
        Returns:
            True if the atom is true according to the oracle, False otherwise
        """
        pass
    
    def is_ground(self, atom: Atom) -> bool:
        """Check if an atom is ground."""
        return atom.is_ground()


class NNOracle(Oracle):
    """
    Oracle for querying feedforward neural network models.
    
    Observation language:
    - predict(instance_id, label): model prediction
    - feature(instance_id, feature_name, value): feature value
    - has_feature(instance_id, feature_name): boolean feature presence
    """
    
    def __init__(
        self,
        model: Any,  # The neural network model
        instances: Dict[int, Dict[str, Any]],  # instance_id -> feature_dict
        feature_names: List[str],
        label_map: Dict[str, Any],  # label_name -> model output value
        feature_value_map: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """
        Initialize the NN Oracle.
        
        Args:
            model: The neural network model with a predict method
            instances: Dictionary mapping instance_id to feature dictionaries
            feature_names: List of feature names
            label_map: Dictionary mapping label names to model output values
            feature_value_map: Optional mapping of feature_name -> value_name -> actual_value
        """
        self.model = model
        self.instances = instances
        self.feature_names = feature_names
        self.label_map = label_map
        self.feature_value_map = feature_value_map or {}
        
        # Cache for predictions
        self._prediction_cache: Dict[int, Any] = {}
    
    def query(self, atom: Atom) -> bool:
        """Query the oracle about a ground atom."""
        if not atom.is_ground():
            raise ValueError(f"Atom {atom} is not ground")
        
        predicate = atom.predicate
        
        if predicate == "predict":
            return self._query_predict(atom)
        elif predicate == "feature":
            return self._query_feature(atom)
        elif predicate == "has_feature":
            return self._query_has_feature(atom)
        else:
            # Unknown predicate - return False
            return False
    
    def _query_predict(self, atom: Atom) -> bool:
        """Query a predict(instance_id, label) atom."""
        if len(atom.arguments) != 2:
            return False
        
        instance_id_arg, label_arg = atom.arguments
        
        # Extract instance_id
        if not isinstance(instance_id_arg, Constant):
            return False
        instance_id = instance_id_arg.value
        
        # Extract label
        if not isinstance(label_arg, Constant):
            return False
        label_name = label_arg.value
        
        # Get model prediction
        if instance_id not in self._prediction_cache:
            if instance_id not in self.instances:
                return False
            instance_features = self.instances[instance_id]
            feature_vector = self._extract_feature_vector(instance_features)
            prediction = self.model.predict(feature_vector)
            self._prediction_cache[instance_id] = prediction
        
        prediction = self._prediction_cache[instance_id]
        
        # Check if prediction matches label
        expected_value = self.label_map.get(label_name)
        if expected_value is None:
            return False
        
        return prediction == expected_value
    
    def _query_feature(self, atom: Atom) -> bool:
        """Query a feature(instance_id, feature_name, value) atom."""
        if len(atom.arguments) != 3:
            return False
        
        instance_id_arg, feature_name_arg, value_arg = atom.arguments
        
        # Extract instance_id
        if not isinstance(instance_id_arg, Constant):
            return False
        instance_id = instance_id_arg.value
        
        # Extract feature_name
        if not isinstance(feature_name_arg, Constant):
            return False
        feature_name = feature_name_arg.value
        
        # Extract value
        if not isinstance(value_arg, Constant):
            return False
        value_name = value_arg.value
        
        # Get instance features
        if instance_id not in self.instances:
            return False
        instance_features = self.instances[instance_id]
        
        # Get actual feature value
        actual_value = instance_features.get(feature_name)
        if actual_value is None:
            return False
        
        # Check if value matches (possibly through value map)
        if feature_name in self.feature_value_map:
            value_map = self.feature_value_map[feature_name]
            mapped_value = value_map.get(value_name)
            if mapped_value is not None:
                return actual_value == mapped_value
        
        # Direct comparison
        return actual_value == value_name
    
    def _query_has_feature(self, atom: Atom) -> bool:
        """Query a has_feature(instance_id, feature_name) atom."""
        if len(atom.arguments) != 2:
            return False
        
        instance_id_arg, feature_name_arg = atom.arguments
        
        # Extract instance_id
        if not isinstance(instance_id_arg, Constant):
            return False
        instance_id = instance_id_arg.value
        
        # Extract feature_name
        if not isinstance(feature_name_arg, Constant):
            return False
        feature_name = feature_name_arg.value
        
        # Check if instance has this feature
        if instance_id not in self.instances:
            return False
        instance_features = self.instances[instance_id]
        return feature_name in instance_features
    
    def _extract_feature_vector(self, instance_features: Dict[str, Any]) -> np.ndarray:
        """Extract feature vector from instance features."""
        vector = []
        for feature_name in self.feature_names:
            value = instance_features.get(feature_name, 0.0)
            if isinstance(value, (int, float)):
                vector.append(float(value))
            else:
                # Convert to numeric if needed
                vector.append(float(hash(str(value)) % 1000))
        return np.array(vector).reshape(1, -1)
    
    def add_instance(self, instance_id: int, features: Dict[str, Any]) -> None:
        """Add a new instance to the oracle."""
        self.instances[instance_id] = features
        # Clear prediction cache for this instance
        if instance_id in self._prediction_cache:
            del self._prediction_cache[instance_id]
    
    def clear_cache(self) -> None:
        """Clear the prediction cache."""
        self._prediction_cache.clear()


