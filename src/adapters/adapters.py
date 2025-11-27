"""Model adapter implementations for different ML frameworks."""

from typing import Any, Optional, Callable
import numpy as np
from .base import ModelAdapter


class GenericAdapter(ModelAdapter):
    """
    Generic adapter for any model with a predict method.
    
    Works with any object that has:
    - predict(X) -> predictions
    - optionally predict_proba(X) -> probabilities
    
    This is the fallback adapter when no specific adapter is detected.
    
    Example:
        >>> adapter = GenericAdapter(my_model)
        >>> prediction = adapter.predict(X[0])
    """
    
    def __init__(
        self,
        model: Any,
        predict_fn: Optional[Callable] = None,
        predict_proba_fn: Optional[Callable] = None
    ):
        """
        Initialize generic adapter.
        
        Args:
            model: Any model object
            predict_fn: Optional custom predict function (default: model.predict)
            predict_proba_fn: Optional custom predict_proba function
        """
        self.model = model
        self._predict_fn = predict_fn or getattr(model, 'predict', None)
        self._predict_proba_fn = predict_proba_fn or getattr(model, 'predict_proba', None)
        
        if self._predict_fn is None:
            raise ValueError("Model must have a predict method or provide predict_fn")
    
    def predict(self, instance: Any) -> Any:
        """Get prediction for instance."""
        if isinstance(instance, np.ndarray) and len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        result = self._predict_fn(instance)
        if isinstance(result, np.ndarray) and result.shape[0] == 1:
            return result[0]
        return result
    
    def predict_proba(self, instance: Any) -> Optional[np.ndarray]:
        """Get prediction probabilities."""
        if self._predict_proba_fn is None:
            return None
        if isinstance(instance, np.ndarray) and len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        result = self._predict_proba_fn(instance)
        if isinstance(result, np.ndarray) and len(result.shape) > 1 and result.shape[0] == 1:
            return result[0]
        return result
    
    @property
    def model_type(self) -> str:
        return "generic"


class SklearnAdapter(ModelAdapter):
    """
    Adapter for scikit-learn models.
    
    Works with any fitted sklearn estimator (classifiers, regressors, etc.).
    
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier().fit(X, y)
        >>> adapter = SklearnAdapter(model)
        >>> prediction = adapter.predict(X[0])
    """
    
    def __init__(self, model: Any):
        """
        Initialize sklearn adapter.
        
        Args:
            model: A fitted scikit-learn estimator
        """
        self.model = model
        if not hasattr(model, 'predict'):
            raise ValueError("Model must have a predict method")
    
    def predict(self, instance: np.ndarray) -> Any:
        """Get prediction for instance."""
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        return self.model.predict(instance)[0]
    
    def predict_proba(self, instance: np.ndarray) -> Optional[np.ndarray]:
        """Get prediction probabilities."""
        if not hasattr(self.model, 'predict_proba'):
            return None
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        return self.model.predict_proba(instance)[0]
    
    @property
    def model_type(self) -> str:
        return "sklearn"


class PyTorchAdapter(ModelAdapter):
    """
    Adapter for PyTorch models.
    
    Works with any nn.Module that outputs class logits.
    
    Example:
        >>> model = MyPyTorchModel()
        >>> model.load_state_dict(torch.load("model.pt"))
        >>> adapter = PyTorchAdapter(model)
        >>> prediction = adapter.predict(X[0])
    """
    
    def __init__(self, model: Any, device: str = "cpu"):
        """
        Initialize PyTorch adapter.
        
        Args:
            model: A PyTorch nn.Module
            device: Device to run inference on ("cpu" or "cuda")
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        # Import torch here to avoid hard dependency
        try:
            import torch
            self.torch = torch
        except ImportError:
            raise ImportError("PyTorch is required for PyTorchAdapter. Install with: pip install torch")
    
    def predict(self, instance: Any) -> int:
        """Get prediction for instance."""
        with self.torch.no_grad():
            if isinstance(instance, np.ndarray):
                instance = self.torch.FloatTensor(instance)
            if len(instance.shape) == 1:
                instance = instance.unsqueeze(0)
            instance = instance.to(self.device)
            output = self.model(instance)
            return self.torch.argmax(output, dim=1).item()
    
    def predict_proba(self, instance: Any) -> np.ndarray:
        """Get prediction probabilities."""
        with self.torch.no_grad():
            if isinstance(instance, np.ndarray):
                instance = self.torch.FloatTensor(instance)
            if len(instance.shape) == 1:
                instance = instance.unsqueeze(0)
            instance = instance.to(self.device)
            output = self.model(instance)
            probs = self.torch.softmax(output, dim=1)
            return probs.cpu().numpy()[0]
    
    @property
    def model_type(self) -> str:
        return "pytorch"


class TensorFlowAdapter(ModelAdapter):
    """
    Adapter for TensorFlow/Keras models.
    
    Works with any Keras model that outputs class probabilities.
    
    Example:
        >>> from tensorflow import keras
        >>> model = keras.models.load_model("model.h5")
        >>> adapter = TensorFlowAdapter(model)
        >>> prediction = adapter.predict(X[0])
    """
    
    def __init__(self, model: Any):
        """
        Initialize TensorFlow adapter.
        
        Args:
            model: A TensorFlow/Keras model
        """
        self.model = model
    
    def predict(self, instance: np.ndarray) -> int:
        """Get prediction for instance."""
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        output = self.model.predict(instance, verbose=0)
        return int(np.argmax(output[0]))
    
    def predict_proba(self, instance: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        output = self.model.predict(instance, verbose=0)
        return output[0]
    
    @property
    def model_type(self) -> str:
        return "tensorflow"


def detect_and_create_adapter(model: Any) -> ModelAdapter:
    """
    Auto-detect model type and create appropriate adapter.
    
    This function inspects the model object and its module to determine
    the best adapter to use. Falls back to GenericAdapter if no specific
    adapter matches.
    
    Args:
        model: Any ML model
        
    Returns:
        Appropriate ModelAdapter instance
        
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier().fit(X, y)
        >>> adapter = detect_and_create_adapter(model)  # Returns SklearnAdapter
    """
    model_module = type(model).__module__
    model_class = type(model).__name__
    
    # Check for PyTorch first (before generic check for predict method)
    # PyTorch models have 'forward' not 'predict'
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            return PyTorchAdapter(model)
    except ImportError:
        pass
    
    # Check for scikit-learn
    if 'sklearn' in model_module:
        return SklearnAdapter(model)
    
    # Check for TensorFlow/Keras
    if 'tensorflow' in model_module or 'keras' in model_module:
        return TensorFlowAdapter(model)
    
    # Check for XGBoost, LightGBM, CatBoost (tree-based)
    if any(name in model_module.lower() for name in ['xgboost', 'lightgbm', 'catboost']):
        return GenericAdapter(model)
    
    # Fallback to generic adapter
    return GenericAdapter(model)

