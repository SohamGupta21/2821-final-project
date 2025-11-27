"""Wrapper to convert neural network models to oracles."""

from typing import Dict, List, Any, Optional
import numpy as np
from ..core.atoms import Atom, Constant
from ..inference.oracle import NNOracle
from .nn_model import SimpleNNClassifier


def wrap_nn(
    model: SimpleNNClassifier,
    instances: Dict[int, Dict[str, Any]],
    feature_names: List[str],
    label_map: Dict[str, int],
    feature_value_map: Optional[Dict[str, Dict[str, float]]] = None
) -> NNOracle:
    """
    Wrap a neural network model as an oracle.
    
    Args:
        model: The trained neural network model
        instances: Dictionary mapping instance_id to feature dictionaries
        feature_names: List of feature names
        label_map: Dictionary mapping label names to model output values
                  (e.g., {"APPROVED": 1, "DENIED": 0})
        feature_value_map: Optional mapping for categorical features
                          (e.g., {"income": {"high": 1.0, "low": -1.0}})
    
    Returns:
        NNOracle instance
    """
    # Create a wrapper that uses the model's predict method
    class ModelWrapper:
        def __init__(self, nn_model):
            self.nn_model = nn_model
        
        def predict(self, feature_vector: np.ndarray) -> int:
            return self.nn_model.predict(feature_vector)
    
    wrapped_model = ModelWrapper(model)
    
    return NNOracle(
        model=wrapped_model,
        instances=instances,
        feature_names=feature_names,
        label_map=label_map,
        feature_value_map=feature_value_map
    )


def create_instances_from_data(
    X: np.ndarray,
    feature_names: List[str],
    discretize: bool = True,
    bins: int = 3
) -> Dict[int, Dict[str, Any]]:
    """
    Create instance dictionaries from feature matrix.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        feature_names: List of feature names
        discretize: Whether to discretize continuous features
        bins: Number of bins for discretization
    
    Returns:
        Dictionary mapping instance_id to feature dictionary
    """
    instances = {}
    n_samples = X.shape[0]
    
    for i in range(n_samples):
        instance_features = {}
        
        for j, feature_name in enumerate(feature_names):
            value = X[i, j]
            
            if discretize:
                # Discretize into bins
                # Simple approach: divide into high/medium/low
                if bins == 3:
                    if value > 0.3:
                        discrete_value = "high"
                    elif value < -0.3:
                        discrete_value = "low"
                    else:
                        discrete_value = "medium"
                else:
                    # Use numeric bin
                    bin_edges = np.linspace(X[:, j].min(), X[:, j].max(), bins + 1)
                    bin_idx = np.digitize(value, bin_edges) - 1
                    discrete_value = f"bin_{bin_idx}"
                
                instance_features[feature_name] = discrete_value
            else:
                instance_features[feature_name] = float(value)
        
        instances[i] = instance_features
    
    return instances


def generate_facts_from_instances(
    instances: Dict[int, Dict[str, Any]],
    model: SimpleNNClassifier,
    feature_names: List[str],
    label_map: Dict[str, int],
    instance_ids: Optional[List[int]] = None
) -> List[Atom]:
    """
    Generate observation facts from instances.
    
    Args:
        instances: Dictionary mapping instance_id to features
        model: The trained model
        feature_names: List of feature names
        label_map: Dictionary mapping label names to model outputs
        instance_ids: Optional list of instance IDs to process
    
    Returns:
        List of ground atoms (facts)
    """
    facts = []
    
    if instance_ids is None:
        instance_ids = list(instances.keys())
    
    for instance_id in instance_ids:
        if instance_id not in instances:
            continue
        
        instance_features = instances[instance_id]
        
        # Extract feature vector
        feature_vector = []
        for feature_name in feature_names:
            value = instance_features.get(feature_name, 0.0)
            if isinstance(value, str):
                # Convert discrete value to numeric (simple hash)
                value = float(hash(value) % 100) / 100.0
            feature_vector.append(float(value))
        
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        # Get model prediction
        prediction = model.predict(feature_vector)
        
        # Find label name for this prediction
        label_name = None
        for name, value in label_map.items():
            if value == prediction:
                label_name = name
                break
        
        if label_name is None:
            continue
        
        # Create predict fact
        facts.append(Atom(
            "predict",
            [Constant(instance_id), Constant(label_name)]
        ))
        
        # Create feature facts
        for feature_name, value in instance_features.items():
            if isinstance(value, str):
                facts.append(Atom(
                    "feature",
                    [Constant(instance_id), Constant(feature_name), Constant(value)]
                ))
            else:
                # For numeric values, we might want to discretize
                facts.append(Atom(
                    "feature",
                    [Constant(instance_id), Constant(feature_name), Constant(float(value))]
                ))
    
    return facts


