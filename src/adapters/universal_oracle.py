"""Universal oracle that works with any model adapter and observation language."""

from typing import Any, Dict, List, Optional
import numpy as np
from .base import ModelAdapter, ObservationLanguage
from ..core.atoms import Atom
from ..inference.oracle import Oracle


class UniversalOracle(Oracle): # this helps us generate facts
    
    def __init__(
        self,
        adapter: ModelAdapter,
        observation_language: ObservationLanguage,
        label_map: Dict[str, Any]
    ):
        """
        Initialize universal oracle.
        
        Args:
            adapter: Model adapter for getting predictions
            observation_language: Language for generating/querying atoms
            label_map: Mapping of label names to model output values
                      e.g., {"approved": 1, "denied": 0}
        """
        self.adapter = adapter
        self.observation_language = observation_language
        self.label_map = label_map
        self.reverse_label_map = {v: k for k, v in label_map.items()}
        
        # Instance storage
        self._instances: Dict[Any, Any] = {}
        self._predictions: Dict[Any, str] = {}  # instance_id -> label_name
        self._prediction_cache: Dict[Any, Any] = {}  # instance_id -> raw prediction
    
    def add_instance(self, instance_id: Any, instance_data: Any) -> None:
        """
        Add an instance to the oracle.
        
        Args:
            instance_id: Unique identifier for the instance
            instance_data: The instance data (format depends on observation language)
        """
        self._instances[instance_id] = instance_data
        # Clear cached prediction for this instance
        if instance_id in self._prediction_cache:
            del self._prediction_cache[instance_id]
        if instance_id in self._predictions:
            del self._predictions[instance_id]
    
    def add_instances(self, instances: Dict[Any, Any]) -> None:
        """
        Add multiple instances.
        
        Args:
            instances: Dictionary mapping instance_id to instance_data
        """
        for instance_id, data in instances.items():
            self.add_instance(instance_id, data)
    
    def get_prediction(self, instance_id: Any) -> Optional[str]:
        """
        Get prediction label for an instance.
        
        Args:
            instance_id: Instance identifier
            
        Returns:
            Label name or None if instance not found
        """
        if instance_id not in self._predictions:
            if instance_id not in self._instances:
                return None
            
            # Get raw prediction from model
            instance_data = self._instances[instance_id]
            prepared_input = self._prepare_input(instance_data)
            raw_pred = self.adapter.predict(prepared_input)
            self._prediction_cache[instance_id] = raw_pred
            
            # Convert to label name
            label_name = self.reverse_label_map.get(raw_pred)
            if label_name is None:
                # Try string conversion as fallback
                label_name = self.reverse_label_map.get(str(raw_pred), str(raw_pred))
            self._predictions[instance_id] = label_name
        
        return self._predictions[instance_id]
    
    def get_prediction_proba(self, instance_id: Any) -> Optional[np.ndarray]:
        """
        Get prediction probabilities for an instance.
        
        Args:
            instance_id: Instance identifier
            
        Returns:
            Probability array or None
        """
        if instance_id not in self._instances:
            return None
        
        instance_data = self._instances[instance_id]
        prepared_input = self._prepare_input(instance_data)
        return self.adapter.predict_proba(prepared_input)
    
    def _prepare_input(self, instance_data: Any) -> Any:
        """
        Prepare instance data for model input.
        
        Converts observation language representation to model input format.
        """
        if isinstance(instance_data, dict):
            # Tabular: convert dict to numpy array
            return np.array(list(instance_data.values()), dtype=np.float32)
        elif isinstance(instance_data, str):
            # Text: return as-is (model adapter handles tokenization)
            return instance_data
        else:
            # Image or other: return as-is
            return instance_data
    
    def generate_facts(self, instance_ids: Optional[List[Any]] = None) -> List[Atom]:
        """
        Generate all facts for instances.
        
        Args:
            instance_ids: Optional list of specific instances (default: all)
            
        Returns:
            List of ground atoms
        """
        if instance_ids is None:
            instance_ids = list(self._instances.keys())
        
        all_facts = []
        for instance_id in instance_ids:
            if instance_id not in self._instances:
                continue
            
            instance_data = self._instances[instance_id]
            prediction = self.get_prediction(instance_id)
            
            if prediction is None:
                continue
            
            # Generate facts using observation language
            facts = self.observation_language.generate_instance_facts(
                instance_id=instance_id,
                instance_data=instance_data,
                prediction=self._prediction_cache.get(instance_id),
                label_name=prediction
            )
            all_facts.extend(facts)
        
        return all_facts
    
    def query(self, atom: Atom) -> bool:
        """
        Query the oracle about a ground atom.
        
        This is the main interface used by the inference algorithm.
        
        Args:
            atom: Ground atom to query
            
        Returns:
            True if atom is true, False otherwise
        """
        if not atom.is_ground():
            raise ValueError(f"Atom {atom} is not ground")
        
        # Ensure predictions are computed for relevant instances
        if atom.arguments:
            instance_id = atom.arguments[0].value
            if instance_id in self._instances and instance_id not in self._predictions:
                self.get_prediction(instance_id)
        
        # Delegate to observation language
        return self.observation_language.query_atom(
            atom, self._instances, self._predictions
        )
    
    def get_instance_count(self) -> int:
        """Return number of instances."""
        return len(self._instances)
    
    def get_instance_ids(self) -> List[Any]:
        """Return all instance IDs."""
        return list(self._instances.keys())
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self._prediction_cache.clear()
        self._predictions.clear()
    
    def clear_all(self) -> None:
        """Clear all instances and caches."""
        self._instances.clear()
        self.clear_cache()


