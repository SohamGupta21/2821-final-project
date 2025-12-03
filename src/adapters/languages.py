
from typing import Any, List, Dict, Optional, Set
import numpy as np
import re
from .base import ObservationLanguage, PredicateSpec
from ..core.atoms import Atom, Constant


def is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating))


class TabularObservationLanguage(ObservationLanguage):

    
    def __init__(
        self,
        feature_names: List[str],
        label_names: List[str],
        discretize: bool = True,
        bins: int = 3,
        bin_labels: Optional[List[str]] = None
    ):

        self.feature_names = feature_names
        self.label_names = label_names
        self.discretize = discretize
        self.bins = bins
        self.bin_labels = bin_labels or ["low", "medium", "high"]
        
        # Feature statistics for discretization (populated by fit_discretizer)
        self._feature_stats: Dict[str, Dict[str, float]] = {}
        
        # Predicate specifications
        self._predicate_specs = [
            PredicateSpec("predict", 2, "Model prediction", ["instance_id", "label"], is_target=True),
            PredicateSpec("feature", 3, "Feature value", ["instance_id", "feature_name", "value"]),
            PredicateSpec("has_feature", 2, "Feature presence", ["instance_id", "feature_name"]),
        ]
    
    def get_predicates(self) -> List[str]:
        return ["predict", "feature", "has_feature"]
    
    def get_predicate_specs(self) -> List[PredicateSpec]:
        return self._predicate_specs
    
    def fit_discretizer(self, X: np.ndarray) -> None:
        for i, name in enumerate(self.feature_names):
            if i >= X.shape[1]:
                break
            col = X[:, i]
            self._feature_stats[name] = {
                "min": float(np.min(col)),
                "max": float(np.max(col)),
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
                "q33": float(np.percentile(col, 33)),
                "q66": float(np.percentile(col, 66)),
            }
    
    def discretize_value(self, feature_name: str, value: float) -> str:
        if feature_name not in self._feature_stats:
            # Fallback: simple discretization based on sign/magnitude
            if value > 0.3:
                return self.bin_labels[-1]  # high
            elif value < -0.3:
                return self.bin_labels[0]   # low
            return self.bin_labels[len(self.bin_labels) // 2]  # medium
        
        stats = self._feature_stats[feature_name]
        if value <= stats["q33"]:
            return self.bin_labels[0]
        elif value >= stats["q66"]:
            return self.bin_labels[-1]
        return self.bin_labels[len(self.bin_labels) // 2]
    
    def generate_instance_facts(
        self,
        instance_id: Any,
        instance_data: Dict[str, Any],
        prediction: Any,
        label_name: str
    ) -> List[Atom]:
        facts = []
        id_const = Constant(instance_id)
        
        # Prediction fact
        facts.append(Atom("predict", [id_const, Constant(label_name)]))
        
        # Feature facts
        for feature_name, value in instance_data.items():
            # Discretize if needed
            if self.discretize and is_numeric(value):
                discrete_value = self.discretize_value(feature_name, float(value))
                facts.append(Atom("feature", [
                    id_const, Constant(feature_name), Constant(discrete_value)
                ]))
            else:
                # Use value as-is (string or already discrete)
                facts.append(Atom("feature", [
                    id_const, Constant(feature_name), Constant(value)
                ]))
            
            # has_feature fact
            facts.append(Atom("has_feature", [id_const, Constant(feature_name)]))
        
        return facts
    
    def query_atom(self, atom: Atom, instances: Dict[Any, Any], predictions: Dict[Any, Any]) -> bool:
        if not atom.is_ground():
            return False
        
        pred = atom.predicate
        args = atom.arguments
        
        if pred == "predict":
            if len(args) != 2:
                return False
            instance_id = args[0].value
            label = args[1].value
            return predictions.get(instance_id) == label
        
        elif pred == "feature":
            if len(args) != 3:
                return False
            instance_id = args[0].value
            feature_name = args[1].value
            expected_value = args[2].value
            
            if instance_id not in instances:
                return False
            instance_data = instances[instance_id]
            actual_value = instance_data.get(feature_name)
            
            if actual_value is None:
                return False
            
            # Discretize if needed for comparison
            if self.discretize and is_numeric(actual_value):
                actual_value = self.discretize_value(feature_name, float(actual_value))
            
            return actual_value == expected_value
        
        elif pred == "has_feature":
            if len(args) != 2:
                return False
            instance_id = args[0].value
            feature_name = args[1].value
            return feature_name in instances.get(instance_id, {})
        
        return False
    
    @property
    def language_name(self) -> str:
        return "tabular"


class ImageObservationLanguage(ObservationLanguage):
    
    def __init__(
        self,
        label_names: List[str],
        image_size: tuple = (28, 28),
        grid_size: int = 4,
        intensity_levels: int = 3
    ):
        self.label_names = label_names
        self.image_size = image_size
        self.grid_size = grid_size
        self.intensity_levels = intensity_levels
        self.intensity_labels = ["low", "medium", "high"][:intensity_levels]
    
    def get_predicates(self) -> List[str]:
        return ["predict", "is_dark", "is_bright", "region_intensity", "has_content"]
    
    def _analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
        features = {}
        
        # Handle different image formats
        if len(image.shape) == 3:
            # Convert to grayscale if color
            if image.shape[2] == 3:
                image = np.mean(image, axis=2)
            else:
                image = image[:, :, 0]
        
        # Normalize to 0-1
        if image.max() > 1:
            image = image / 255.0
        
        # Overall intensity
        mean_intensity = np.mean(image)
        features["is_dark"] = mean_intensity < 0.3
        features["is_bright"] = mean_intensity > 0.7
        features["has_content"] = np.std(image) > 0.1  # Has meaningful variation
        
        # Region analysis
        h, w = image.shape[:2]
        region_h = max(1, h // self.grid_size)
        region_w = max(1, w // self.grid_size)
        features["regions"] = {}
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                y_start, y_end = i * region_h, min((i + 1) * region_h, h)
                x_start, x_end = j * region_w, min((j + 1) * region_w, w)
                
                region = image[y_start:y_end, x_start:x_end]
                if region.size == 0:
                    continue
                    
                region_mean = np.mean(region)
                region_name = f"r{i}_{j}"
                
                # Classify intensity
                if region_mean < 0.33:
                    features["regions"][region_name] = "low"
                elif region_mean > 0.66:
                    features["regions"][region_name] = "high"
                else:
                    features["regions"][region_name] = "medium"
        
        return features
    
    def generate_instance_facts(
        self,
        instance_id: Any,
        instance_data: np.ndarray,
        prediction: Any,
        label_name: str
    ) -> List[Atom]:
        facts = []
        id_const = Constant(instance_id)
        
        # Prediction
        facts.append(Atom("predict", [id_const, Constant(label_name)]))
        
        # Analyze image
        analysis = self._analyze_image(instance_data)
        
        # Global intensity facts
        if analysis["is_dark"]:
            facts.append(Atom("is_dark", [id_const]))
        if analysis["is_bright"]:
            facts.append(Atom("is_bright", [id_const]))
        if analysis["has_content"]:
            facts.append(Atom("has_content", [id_const]))
        
        # Region intensity facts
        for region_name, intensity in analysis["regions"].items():
            facts.append(Atom("region_intensity", [
                id_const, Constant(region_name), Constant(intensity)
            ]))
        
        return facts
    
    def query_atom(self, atom: Atom, instances: Dict[Any, Any], predictions: Dict[Any, Any]) -> bool:
        if not atom.is_ground():
            return False
        
        pred = atom.predicate
        args = atom.arguments
        
        if pred == "predict":
            if len(args) != 2:
                return False
            instance_id = args[0].value
            label = args[1].value
            return predictions.get(instance_id) == label
        
        # Get instance
        if len(args) < 1:
            return False
        instance_id = args[0].value
        if instance_id not in instances:
            return False
        
        # Analyze image
        analysis = self._analyze_image(instances[instance_id])
        
        if pred == "is_dark":
            return analysis["is_dark"]
        elif pred == "is_bright":
            return analysis["is_bright"]
        elif pred == "has_content":
            return analysis.get("has_content", False)
        elif pred == "region_intensity":
            if len(args) != 3:
                return False
            region = args[1].value
            expected_intensity = args[2].value
            return analysis["regions"].get(region) == expected_intensity
        
        return False
    
    @property
    def language_name(self) -> str:
        return "image"


class TextObservationLanguage(ObservationLanguage):
    
    def __init__(
        self,
        label_names: List[str],
        vocabulary: Optional[Set[str]] = None,
        max_vocab_size: int = 1000,
        min_word_freq: int = 2
    ):
        self.label_names = label_names
        self.vocabulary = vocabulary or set()
        self.max_vocab_size = max_vocab_size
        self.min_word_freq = min_word_freq
        
        # Word frequency counts for building vocabulary
        self._word_counts: Dict[str, int] = {}
    
    def get_predicates(self) -> List[str]:
        """Return list of predicates."""
        return ["predict", "has_word", "word_count_level", "is_short", "is_long"]
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())
    
    def update_vocabulary(self, tokens: List[str]) -> None:
        for token in tokens:
            self._word_counts[token] = self._word_counts.get(token, 0) + 1
        
        # Rebuild vocabulary if needed
        if len(self.vocabulary) < self.max_vocab_size:
            # Add frequent words
            sorted_words = sorted(
                self._word_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            for word, count in sorted_words[:self.max_vocab_size]:
                if count >= self.min_word_freq:
                    self.vocabulary.add(word)
    
    def generate_instance_facts(
        self,
        instance_id: Any,
        instance_data: str,
        prediction: Any,
        label_name: str
    ) -> List[Atom]:
        facts = []
        id_const = Constant(instance_id)
        
        # Prediction
        facts.append(Atom("predict", [id_const, Constant(label_name)]))
        
        # Tokenize
        tokens = self._tokenize(instance_data)
        unique_tokens = set(tokens)
        
        # Update vocabulary
        self.update_vocabulary(tokens)
        
        # Word presence facts (only for words in vocabulary, limit to prevent explosion)
        vocab_words = unique_tokens.intersection(self.vocabulary)
        for word in list(vocab_words)[:50]:
            facts.append(Atom("has_word", [id_const, Constant(word)]))
        
        # Length facts
        word_count = len(tokens)
        if word_count < 50:
            facts.append(Atom("is_short", [id_const]))
        if word_count > 200:
            facts.append(Atom("is_long", [id_const]))
        
        # Word count level
        if word_count < 20:
            level = "very_short"
        elif word_count < 100:
            level = "short"
        elif word_count < 300:
            level = "medium"
        else:
            level = "long"
        facts.append(Atom("word_count_level", [id_const, Constant(level)]))
        
        return facts
    
    def query_atom(self, atom: Atom, instances: Dict[Any, Any], predictions: Dict[Any, Any]) -> bool:
        if not atom.is_ground():
            return False
        
        pred = atom.predicate
        args = atom.arguments
        
        if pred == "predict":
            if len(args) != 2:
                return False
            instance_id = args[0].value
            label = args[1].value
            return predictions.get(instance_id) == label
        
        # Get instance
        if len(args) < 1:
            return False
        instance_id = args[0].value
        if instance_id not in instances:
            return False
        
        text = instances[instance_id]
        tokens = self._tokenize(text)
        
        if pred == "has_word":
            if len(args) != 2:
                return False
            word = args[1].value
            return word in tokens
        elif pred == "is_short":
            return len(tokens) < 50
        elif pred == "is_long":
            return len(tokens) > 200
        elif pred == "word_count_level":
            if len(args) != 2:
                return False
            level = args[1].value
            word_count = len(tokens)
            actual_level = (
                "very_short" if word_count < 20 else
                "short" if word_count < 100 else
                "medium" if word_count < 300 else "long"
            )
            return actual_level == level
        
        return False
    
    @property
    def language_name(self) -> str:
        return "text"

