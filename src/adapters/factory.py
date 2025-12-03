"""Factory functions for easy plug-and-play explainability."""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import numpy as np

from .base import ModelAdapter, ObservationLanguage
from .adapters import detect_and_create_adapter
from .languages import TabularObservationLanguage, ImageObservationLanguage, TextObservationLanguage
from .universal_oracle import UniversalOracle
from ..inference.algorithm import ModelInference
from ..core.theory import Theory
from ..core.clauses import Clause
from ..core.atoms import Constant, Variable


@dataclass
class ExplainabilityResult:
    """
    Container for explainability results.
    
    Provides easy access to learned rules, metrics, and explanations.
    This is the main output of the create_explainer function.
    
    Attributes:
        theory: The learned theory (collection of clauses)
        history: Evolution of the theory during learning
        oracle: The oracle used for queries
        metrics: Computed explainability metrics
    
    Example:
        >>> result = create_explainer(model, X)
        >>> print(result.summary())
        >>> for rule in result.rules:
        ...     print(rule)
    """
    theory: Theory
    history: List[Theory]
    oracle: UniversalOracle
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def rules(self) -> List[Clause]:
        """Get learned rules (clauses with body)."""
        return self.theory.get_rules()
    
    @property
    def facts(self) -> List[Clause]:
        """Get learned facts (clauses without body)."""
        return self.theory.get_facts()
    
    @property
    def num_rules(self) -> int:
        """Number of learned rules."""
        return len(self.rules)
    
    @property
    def num_clauses(self) -> int:
        """Total number of clauses."""
        return len(self.theory)
    
    def get_rules_for_label(self, label: str) -> List[Clause]:
        """
        Get rules that predict a specific label.
        
        Args:
            label: The label to filter by
            
        Returns:
            List of clauses that predict this label
        """
        rules = []
        for clause in self.theory.get_clauses():
            head = clause.head
            if head.predicate == "predict" and len(head.arguments) >= 2:
                # Check if second argument matches label
                label_arg = head.arguments[1]
                if isinstance(label_arg, Constant):
                    if label_arg.value == label:
                        rules.append(clause)
                elif str(label_arg) == label:
                    rules.append(clause)
        return rules
    
    def explain_prediction(self, instance_id: Any) -> Dict[str, Any]:
        """
        Explain why the model made a prediction for an instance.
        
        This finds all rules that apply to the given instance and
        extracts the key features that triggered those rules.
        
        Args:
            instance_id: The instance to explain
            
        Returns:
            Dictionary with:
            - prediction: The predicted label
            - matching_rules: Rules that apply to this instance
            - key_features: Features that triggered the rules
            - confidence: Number of matching rules (more = higher confidence)
        """
        prediction = self.oracle.get_prediction(instance_id)
        if prediction is None:
            return {
                "instance_id": instance_id,
                "prediction": None,
                "matching_rules": [],
                "key_features": [],
                "num_matching_rules": 0,
                "error": "Instance not found"
            }
        
        matching_rules = []
        key_features = set()
        
        for rule in self.get_rules_for_label(prediction):
            # Check if rule body is satisfied for this instance
            body_satisfied = True
            rule_features = []
            
            for body_atom in rule.body:
                # Substitute instance_id into body atom
                # The rule uses variable X for instance_id
                substitution = {Variable("X"): Constant(instance_id)}
                substituted = body_atom.apply_substitution(substitution)
                
                if not self.oracle.query(substituted):
                    body_satisfied = False
                    break
                
                # Extract feature info from body atom
                if body_atom.predicate == "feature" and len(body_atom.arguments) >= 3:
                    feature_name = body_atom.arguments[1]
                    feature_value = body_atom.arguments[2]
                    if isinstance(feature_name, Constant) and isinstance(feature_value, Constant):
                        rule_features.append(f"{feature_name.value}={feature_value.value}")
                    elif isinstance(feature_name, Constant):
                        rule_features.append(f"{feature_name.value}={feature_value}")
            
            if body_satisfied:
                matching_rules.append(rule)
                key_features.update(rule_features)
        
        return {
            "instance_id": instance_id,
            "prediction": prediction,
            "matching_rules": matching_rules,
            "key_features": list(key_features),
            "num_matching_rules": len(matching_rules)
        }
    
    def compute_metrics(self, test_instances: Optional[List[Any]] = None) -> Dict[str, float]:
        """
        Compute explainability metrics.
        
        Args:
            test_instances: Optional list of instance IDs to evaluate on
            
        Returns:
            Dictionary with:
            - rule_coverage: Fraction of predictions explained by rules
            - avg_rule_length: Average conditions per rule
            - interpretability_score: Composite metric (higher = more interpretable)
            - theory_size: Total number of clauses
        """
        if test_instances is None:
            test_instances = self.oracle.get_instance_ids()
        
        if not test_instances:
            self.metrics = {
                "rule_coverage": 0.0,
                "avg_rule_length": 0.0,
                "num_rules": 0,
                "num_clauses": 0,
                "interpretability_score": 0.0
            }
            return self.metrics
        
        # Coverage: how many predictions are explained by at least one rule
        explained = 0
        for instance_id in test_instances:
            explanation = self.explain_prediction(instance_id)
            if explanation["num_matching_rules"] > 0:
                explained += 1
        
        rule_coverage = explained / len(test_instances)
        
        # Average rule length (number of conditions)
        rule_lengths = [len(rule.body) for rule in self.rules]
        avg_rule_length = float(np.mean(rule_lengths)) if rule_lengths else 0.0
        
        # Interpretability score
        # Shorter rules are more interpretable
        # More coverage is better
        # Formula: coverage * (1 / (1 + avg_length * 0.2))
        if self.rules:
            length_factor = 1.0 / (1.0 + avg_rule_length * 0.2)
            interpretability_score = rule_coverage * length_factor
        else:
            interpretability_score = 0.0
        
        self.metrics = {
            "rule_coverage": rule_coverage,
            "avg_rule_length": avg_rule_length,
            "num_rules": len(self.rules),
            "num_clauses": len(self.theory),
            "interpretability_score": interpretability_score,
            "num_instances": len(test_instances),
            "num_explained": explained
        }
        
        return self.metrics
    
    def summary(self) -> str:
        """
        Generate a human-readable summary.
        
        Returns:
            Formatted string with metrics and learned rules
        """
        if not self.metrics:
            self.compute_metrics()
        
        lines = [
            "",
            "=" * 60,
            "EXPLAINABILITY RESULTS",
            "=" * 60,
            "",
            f"Model type: {self.oracle.adapter.model_type}",
            f"Observation language: {self.oracle.observation_language.language_name}",
            f"Instances analyzed: {self.metrics.get('num_instances', 'N/A')}",
            "",
            "METRICS:",
            "-" * 40,
            f"  Total clauses learned: {self.metrics['num_clauses']}",
            f"  Rules learned: {self.metrics['num_rules']}",
            f"  Average rule length: {self.metrics['avg_rule_length']:.2f} conditions",
            f"  Rule coverage: {self.metrics['rule_coverage']*100:.1f}%",
            f"  Interpretability score: {self.metrics['interpretability_score']:.3f}",
            "",
            "LEARNED RULES:",
            "-" * 40,
        ]
        
        if not self.rules:
            lines.append("  (No rules learned)")
        else:
            # Show rules, sorted by body length (shorter = simpler)
            sorted_rules = sorted(self.rules, key=lambda r: len(r.body))
            for i, rule in enumerate(sorted_rules[:15], 1):  # Show top 15
                lines.append(f"  {i}. {rule}")
            
            if len(self.rules) > 15:
                lines.append(f"     ... and {len(self.rules) - 15} more rules")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def __str__(self) -> str:
        return self.summary()
    
    def __repr__(self) -> str:
        return f"ExplainabilityResult(rules={self.num_rules}, clauses={self.num_clauses})"


def create_explainer(
    model: Any,
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    label_names: Optional[List[str]] = None,
    data_type: str = "tabular",
    max_instances: int = 100,
    verbose: bool = True,
    **kwargs
) -> ExplainabilityResult:
    """
    Create an explainer for any model with minimal configuration.
    
    This is the main entry point for plug-and-play explainability.
    Simply pass your trained model and data, and get interpretable rules.
    
    Args:
        model: Any trained ML model (sklearn, pytorch, tensorflow, etc.)
        X: Input data (features)
        y: Optional ground truth labels (for validation)
        feature_names: Names for features (auto-generated if not provided)
        label_names: Names for class labels (auto-generated if not provided)
        data_type: Type of data ("tabular", "image", "text")
        max_instances: Maximum instances to use for inference
        verbose: Whether to print progress
        **kwargs: Additional arguments:
            - discretize: Whether to discretize features (default: True)
            - bins: Number of bins for discretization (default: 3)
            - max_iterations: Max iterations per fact (default: 50)
            - max_theory_size: Max clauses in theory (default: 30)
            - image_size: Expected image size (default: (28, 28))
            - grid_size: Grid divisions for images (default: 4)
    
    Returns:
        ExplainabilityResult with learned rules and metrics
    
    Examples:
        # Sklearn model
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier().fit(X_train, y_train)
        >>> result = create_explainer(model, X_test, feature_names=["age", "income"])
        >>> print(result.summary())
        
        # PyTorch model
        >>> result = create_explainer(pytorch_model, X_test, label_names=["cat", "dog"])
        
        # One-liner
        >>> print(explain_model(model, X))
    """
    if verbose:
        print("Setting up explainability analysis...")
    
    # Auto-detect and create adapter
    adapter = detect_and_create_adapter(model)
    if verbose:
        print(f"  Detected model type: {adapter.model_type}")
    
    # Generate feature names if not provided
    if feature_names is None:
        if data_type == "tabular" and len(X.shape) > 1:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        else:
            feature_names = []
    
    # Determine labels
    if label_names is None:
        # Try to get from model
        if hasattr(model, 'classes_'):
            unique_labels = list(model.classes_)
        elif y is not None:
            unique_labels = sorted(set(y))
        else:
            # Predict on sample to find labels
            sample_preds = set()
            sample_size = min(50, len(X))
            for i in range(sample_size):
                try:
                    pred = adapter.predict(X[i] if len(X.shape) > 1 else X[i:i+1])
                    sample_preds.add(pred)
                except:
                    pass
            unique_labels = sorted(sample_preds)
        
        label_names = [f"class_{l}" for l in unique_labels]
        if verbose:
            print(f"  Auto-detected {len(label_names)} classes")
    
    # Create label map
    if hasattr(model, 'classes_'):
        label_map = {name: cls for name, cls in zip(label_names, model.classes_)}
    else:
        # Assume labels are 0, 1, 2, ...
        label_map = {name: i for i, name in enumerate(label_names)}
    
    if verbose:
        print(f"  Label mapping: {label_map}")
    
    # Create observation language based on data type
    if data_type == "tabular":
        obs_lang = TabularObservationLanguage(
            feature_names=feature_names,
            label_names=label_names,
            discretize=kwargs.get("discretize", True),
            bins=kwargs.get("bins", 3)
        )
        # Fit discretizer on data
        obs_lang.fit_discretizer(X)
        if verbose:
            print(f"  Using tabular observation language with {len(feature_names)} features")
    
    elif data_type == "image":
        obs_lang = ImageObservationLanguage(
            label_names=label_names,
            image_size=kwargs.get("image_size", (28, 28)),
            grid_size=kwargs.get("grid_size", 4)
        )
        if verbose:
            print(f"  Using image observation language")
    
    elif data_type == "text":
        obs_lang = TextObservationLanguage(
            label_names=label_names,
            vocabulary=kwargs.get("vocabulary"),
            max_vocab_size=kwargs.get("max_vocab_size", 1000)
        )
        if verbose:
            print(f"  Using text observation language")
    
    else:
        raise ValueError(f"Unknown data type: {data_type}. Use 'tabular', 'image', or 'text'.")
    
    # Create oracle
    oracle = UniversalOracle(adapter, obs_lang, label_map)
    
    # Add instances
    n_instances = min(max_instances, len(X))
    if verbose:
        print(f"  Adding {n_instances} instances...")
    
    for i in range(n_instances):
        if data_type == "tabular":
            # Convert to dict for tabular data
            instance_data = dict(zip(feature_names, X[i]))
        else:
            # Use raw data for image/text
            instance_data = X[i]
        oracle.add_instance(i, instance_data)
    
    # Generate facts
    if verbose:
        print("  Generating observation facts...")
    all_facts = oracle.generate_facts()
    predict_facts = [f for f in all_facts if f.predicate == "predict"]
    feature_facts = [f for f in all_facts if f.predicate == "feature"]
    
    if verbose:
        print(f"  Generated {len(all_facts)} facts ({len(predict_facts)} predictions, {len(feature_facts)} features)")
    
    # Run inference
    if verbose:
        print("  Running model inference algorithm...")
    
    inference = ModelInference(
        oracle=oracle,
        max_iterations=kwargs.get("max_iterations", 50),
        max_theory_size=kwargs.get("max_theory_size", 30),
        min_support=kwargs.get("min_support", 0.1),
        min_confidence=kwargs.get("min_confidence", 0.6),
        verbose=kwargs.get("algorithm_verbose", False)
    )
    
    # Pass ALL facts (not just predict) so algorithm can analyze patterns
    theory = inference.infer_theory(iter(all_facts))
    history = inference.get_history()
    
    if verbose:
        print(f"  Learned {len(theory)} clauses ({len(theory.get_rules())} rules)")
    
    # Create result
    result = ExplainabilityResult(
        theory=theory,
        history=history,
        oracle=oracle
    )
    
    # Compute metrics
    if verbose:
        print("  Computing metrics...")
    result.compute_metrics(list(range(n_instances)))
    
    if verbose:
        print("Done!")
    
    return result


def explain_model(model: Any, X: np.ndarray, **kwargs) -> str:
    """
    One-liner to explain any model.
    
    This is the simplest way to get explainability - just pass your model
    and data, and get a human-readable explanation.
    
    Args:
        model: Any trained ML model
        X: Input data
        **kwargs: Additional arguments passed to create_explainer
    
    Returns:
        Human-readable explanation string
    
    Example:
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> model = DecisionTreeClassifier().fit(X_train, y_train)
        >>> print(explain_model(model, X_test))
    """
    # Default to non-verbose for one-liner
    kwargs.setdefault("verbose", False)
    result = create_explainer(model, X, **kwargs)
    return result.summary()

