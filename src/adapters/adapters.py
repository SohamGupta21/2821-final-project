"""Model adapter implementations for different ML frameworks."""

from typing import Any, Optional, Callable
import numpy as np
from .base import ModelAdapter


class GenericAdapter(ModelAdapter):
    
    def __init__(
        self,
        model: Any,
        predict_fn: Optional[Callable] = None,
        predict_proba_fn: Optional[Callable] = None
    ):
        self.model = model
        self._predict_fn = predict_fn or getattr(model, 'predict', None)
        self._predict_proba_fn = predict_proba_fn or getattr(model, 'predict_proba', None)
        
        if self._predict_fn is None:
            raise ValueError("Model must have a predict method or provide predict_fn")
    
    def predict(self, instance: Any) -> Any:
        if isinstance(instance, np.ndarray) and len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        result = self._predict_fn(instance)
        if isinstance(result, np.ndarray) and result.shape[0] == 1:
            return result[0]
        return result
    
    def predict_proba(self, instance: Any) -> Optional[np.ndarray]:
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
    
    def __init__(self, model: Any):
        self.model = model
        if not hasattr(model, 'predict'):
            raise ValueError("Model must have a predict method")
    
    def predict(self, instance: np.ndarray) -> Any:
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        return self.model.predict(instance)[0]
    
    def predict_proba(self, instance: np.ndarray) -> Optional[np.ndarray]:
        if not hasattr(self.model, 'predict_proba'):
            return None
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        return self.model.predict_proba(instance)[0]
    
    @property
    def model_type(self) -> str:
        return "sklearn"


class PyTorchAdapter(ModelAdapter):
    
    def __init__(self, model: Any, device: str = "cpu"):
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
        with self.torch.no_grad():
            if isinstance(instance, np.ndarray):
                instance = self.torch.FloatTensor(instance)
            if len(instance.shape) == 1:
                instance = instance.unsqueeze(0)
            instance = instance.to(self.device)
            output = self.model(instance)
            return self.torch.argmax(output, dim=1).item()
    
    def predict_proba(self, instance: Any) -> np.ndarray:
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
    
    def __init__(self, model: Any):
        self.model = model
    
    def predict(self, instance: np.ndarray) -> int:
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        output = self.model.predict(instance, verbose=0)
        return int(np.argmax(output[0]))
    
    def predict_proba(self, instance: np.ndarray) -> np.ndarray:
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        output = self.model.predict(instance, verbose=0)
        return output[0]
    
    @property
    def model_type(self) -> str:
        return "tensorflow"


def detect_and_create_adapter(model: Any) -> ModelAdapter:
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

