# Shapiro's Model Inference for Neural Network Explainability

This project implements **Ehud Shapiro's Model Inference Algorithm (1981)** as an explainability system for black-box machine learning models. The system infers logical Horn clause rules that explain model behavior by learning interpretable patterns from model predictions and input features.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Complete Architecture Documentation](#complete-architecture-documentation)
5. [Detailed Module Reference](#detailed-module-reference)
6. [API Reference](#api-reference)
7. [How It Works: Step-by-Step](#how-it-works-step-by-step)
8. [Examples](#examples)
9. [Extending the System](#extending-the-system)
10. [Testing](#testing)
11. [Visualization](#visualization)

---

## Overview

### What This System Does

This system takes any trained machine learning model (neural network, random forest, SVM, etc.) and automatically learns **interpretable logical rules** that explain how the model makes predictions. Instead of treating the model as a black box, you get human-readable rules like:

```
predict(X, approved) :- feature(X, income, high), feature(X, credit, high).
predict(X, denied) :- feature(X, credit, low).
```

### Key Features

- **Plug-and-Play API**: Explain any model with minimal configuration
- **Framework Agnostic**: Works with scikit-learn, PyTorch, TensorFlow, and any model with a `predict()` method
- **Multiple Data Types**: Tabular, image, and text data support
- **Interpretable Output**: Human-readable logical Horn clause rules
- **Comprehensive Metrics**: Rule coverage, interpretability scores, and prediction explanations
- **Visualization Tools**: Proof tree visualization, theory evolution plots, and rule coverage analysis

---

## Installation

### Prerequisites
- Python 3.7 or higher
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Development Setup

For development, you may want to install the package in editable mode:

```bash
pip install -e .
```

### Dependencies
- `torch>=2.0.0` - PyTorch for neural network models
- `numpy>=1.24.0` - Numerical computations
- `matplotlib>=3.7.0` - Visualization
- `scikit-learn>=1.3.0` - Machine learning utilities
- `pandas>=2.0.0` - Data manipulation

---

## Quick Start

### One-Liner Explanation

```python
from src import explain_model

# Explain any trained model!
print(explain_model(model, X))
```

### Detailed Analysis

```python
from src import create_explainer

# Create explainer with full control
result = create_explainer(
    model=model,
    X=X_test,
    feature_names=["age", "income", "credit_score"],
    label_names=["denied", "approved"]
)

# View summary
print(result.summary())

# Access learned rules
for rule in result.rules:
    print(rule)

# Explain specific predictions
explanation = result.explain_prediction(instance_id=0)
print(f"Prediction: {explanation['prediction']}")
print(f"Key features: {explanation['key_features']}")

# Get metrics
metrics = result.compute_metrics()
print(f"Rule coverage: {metrics['rule_coverage']*100:.1f}%")
```

---

## Complete Architecture Documentation

### System Architecture Overview

The system is organized into five main modules:

1. **`src/core/`** - Core logical reasoning components (atoms, clauses, theories, resolution)
2. **`src/adapters/`** - Model adapters and observation languages (interface between ML models and logical representation)
3. **`src/inference/`** - Model inference algorithm (Shapiro's algorithm implementation)
4. **`src/models/`** - Neural network models and utilities
5. **`src/utils/`** - Visualization and utility functions

### Data Flow

```
User Input (model, X)
    ↓
Model Adapter (wraps model with unified interface)
    ↓
Observation Language (converts data to logical atoms)
    ↓
Universal Oracle (generates facts and answers queries)
    ↓
Model Inference Algorithm (learns rules from facts)
    ↓
Theory (collection of learned clauses)
    ↓
ExplainabilityResult (rules + metrics + explanations)
```

---

## Detailed Module Reference

### Module: `src/core/` - Core Logical Reasoning

This module provides the foundational data structures for logical reasoning: atoms, terms, clauses, theories, and resolution.

#### File: `src/core/atoms.py`

**Purpose**: Defines logical atoms, terms (variables/constants), and unification algorithm.

##### Classes

**`Term` (Abstract Base Class)**
- Base class for all logical terms (variables, constants, structured terms)
- **Methods**:
  - `__str__() -> str`: String representation
  - `__eq__(other) -> bool`: Equality comparison
  - `__hash__() -> int`: Hash for use in sets/dicts
  - `apply_substitution(substitution: Dict[Variable, Term]) -> Term`: Apply variable substitution
  - `is_variable() -> bool`: Check if term is a variable

**`Variable(Term)`**
- Represents a logical variable (e.g., `X`, `Y`)
- **Attributes**:
  - `name: str`: Variable name
- **Methods**: Inherits all from `Term`

**`Constant(Term)`**
- Represents a constant term (e.g., `"approved"`, `42`, `"high"`)
- **Attributes**:
  - `value: Any`: Constant value (can be any Python object)
- **Methods**: Inherits all from `Term`

**`Atom`**
- Represents a logical atom: a predicate with arguments (e.g., `predict(X, approved)`)
- **Attributes**:
  - `predicate: str`: Predicate name (e.g., `"predict"`, `"feature"`)
  - `arguments: List[Term]`: List of terms as arguments
- **Methods**:
  - `__str__() -> str`: String representation (e.g., `"predict(X, approved)"`)
  - `__eq__(other) -> bool`: Equality comparison
  - `__hash__() -> int`: Hash for use in sets/dicts
  - `is_ground() -> bool`: Check if atom contains no variables
  - `apply_substitution(substitution: Dict[Variable, Term]) -> Atom`: Apply substitution to all terms
  - `get_variables() -> List[Variable]`: Extract all variables from arguments

##### Functions

**`unify(term1: Term, term2: Term, substitution: Optional[Dict[Variable, Term]] = None) -> Optional[Dict[Variable, Term]]`**
- Unify two terms, returning a substitution that makes them equal
- Returns `None` if unification is impossible
- Implements the standard unification algorithm with occurs check

**`unify_atoms(atom1: Atom, atom2: Atom) -> Optional[Dict[Variable, Term]]`**
- Unify two atoms (must have same predicate and arity)
- Returns substitution if atoms can be unified, `None` otherwise

**`compose_substitutions(sub1: Dict[Variable, Term], sub2: Dict[Variable, Term]) -> Dict[Variable, Term]`**
- Compose two substitutions: `sub1 o sub2` (apply sub2 first, then sub1)

#### File: `src/core/clauses.py`

**Purpose**: Defines Horn clauses (head :- body).

##### Classes

**`Clause`**
- Represents a Horn clause: `head :- body`
- **Attributes**:
  - `head: Atom`: The head atom (conclusion)
  - `body: List[Atom]`: List of body atoms (conditions)
- **Methods**:
  - `__str__() -> str`: String representation (e.g., `"predict(X, approved) :- feature(X, income, high)."`)
  - `is_fact() -> bool`: Check if clause is a fact (empty body)
  - `is_rule() -> bool`: Check if clause is a rule (non-empty body)
  - `apply_substitution(substitution: Dict[Variable, Term]) -> Clause`: Apply substitution to all atoms
  - `get_variables() -> List[Variable]`: Get all variables in the clause
  - `is_ground() -> bool`: Check if clause is ground (no variables)
  - `standardize_apart(used_variables: Optional[List[Variable]] = None) -> Clause`: Rename variables to avoid conflicts
  - `pretty_print(indent: int = 0) -> str`: Pretty-print with indentation

#### File: `src/core/theory.py`

**Purpose**: Manages a collection of clauses (a theory).

##### Classes

**`Theory`**
- A theory is a collection of Horn clauses
- **Methods**:
  - `__init__(clauses: Optional[List[Clause]] = None)`: Initialize with optional list of clauses
  - `add_clause(clause: Clause) -> None`: Add a clause (duplicates are ignored)
  - `remove_clause(clause: Clause) -> bool`: Remove a clause, returns True if removed
  - `get_clauses() -> List[Clause]`: Get all clauses
  - `get_facts() -> List[Clause]`: Get all fact clauses (no body)
  - `get_rules() -> List[Clause]`: Get all rule clauses (with body)
  - `__len__() -> int`: Number of clauses
  - `__contains__(clause: Clause) -> bool`: Check if clause is in theory
  - `__iter__()`: Iterate over clauses
  - `copy() -> Theory`: Create a copy
  - `clear() -> None`: Remove all clauses
  - `find_clauses_with_head(atom: Atom) -> List[Clause]`: Find clauses whose head unifies with atom

#### File: `src/core/resolution.py`

**Purpose**: SLD resolution engine for logical inference and proof generation.

##### Classes

**`ProofNode`**
- Represents a node in a resolution proof tree
- **Attributes**:
  - `goal: Atom`: The goal atom being proven
  - `substitution: Dict[Variable, Term]`: Current variable substitution
  - `clause: Optional[Clause]`: Clause used to resolve this goal
  - `parent: Optional[ProofNode]`: Parent node in tree
  - `children: List[ProofNode]`: Child nodes
  - `depth: int`: Depth in proof tree
- **Methods**:
  - `is_leaf() -> bool`: Check if node has no children
  - `is_root() -> bool`: Check if node has no parent
  - `get_path_to_root() -> List[ProofNode]`: Get path from this node to root
  - `get_all_leaf_clauses() -> List[Clause]`: Get all clauses used at leaf nodes

**`ResolutionEngine`**
- SLD resolution engine for proving goals from a theory
- **Methods**:
  - `__init__(max_depth: int = 10)`: Initialize with maximum proof depth
  - `prove(theory: Theory, goal: Atom, oracle: Optional[Callable[[Atom], bool]] = None) -> Optional[ProofNode]`: Attempt to prove goal, returns proof tree if successful
  - `find_contradiction(theory: Theory, fact: Atom, oracle: Callable[[Atom], bool]) -> Optional[ProofNode]`: Find contradiction (theory proves fact but oracle says false)
  - `can_prove(theory: Theory, goal: Atom, oracle: Optional[Callable[[Atom], bool]] = None) -> bool`: Check if goal can be proven

---

### Module: `src/adapters/` - Model Adapters and Observation Languages

This module provides the interface between machine learning models and the logical reasoning system.

#### File: `src/adapters/base.py`

**Purpose**: Abstract base classes for model adapters and observation languages.

##### Classes

**`ModelAdapter` (Abstract Base Class)**
- Abstract interface for wrapping ML models
- **Methods** (all abstract):
  - `predict(instance: Any) -> Any`: Get model prediction for an instance
  - `predict_proba(instance: Any) -> Optional[np.ndarray]`: Get prediction probabilities (optional)
  - `model_type: str` (property): Return model type identifier

**`ObservationLanguage` (Abstract Base Class)**
- Abstract interface for converting data to logical atoms
- **Methods** (all abstract):
  - `get_predicates() -> List[str]`: Return list of predicate names used
  - `generate_instance_facts(instance_id: Any, instance_data: Any, prediction: Any, label_name: str) -> List[Atom]`: Generate facts for an instance
  - `query_atom(atom: Atom, instances: Dict[Any, Any], predictions: Dict[Any, Any]) -> bool`: Check if atom is true for given instances
  - `language_name: str` (property): Return language name

**`PredicateSpec`**
- Specification for a predicate (metadata)
- **Attributes**:
  - `name: str`: Predicate name
  - `arity: int`: Number of arguments
  - `description: str`: Human-readable description
  - `argument_names: List[str]`: Names of arguments
  - `is_target: bool`: Whether this is a target predicate (for predictions)

#### File: `src/adapters/adapters.py`

**Purpose**: Concrete implementations of model adapters for different ML frameworks.

##### Classes

**`GenericAdapter(ModelAdapter)`**
- Fallback adapter for any model with a `predict()` method
- **Methods**:
  - `__init__(model: Any, predict_fn: Optional[Callable] = None, predict_proba_fn: Optional[Callable] = None)`: Initialize with model and optional custom functions
  - `predict(instance: Any) -> Any`: Call model's predict method
  - `predict_proba(instance: Any) -> Optional[np.ndarray]`: Call model's predict_proba if available
  - `model_type: str` (property): Returns `"generic"`

**`SklearnAdapter(ModelAdapter)`**
- Adapter for scikit-learn models
- **Methods**:
  - `__init__(model: Any)`: Initialize with sklearn model
  - `predict(instance: np.ndarray) -> Any`: Get prediction (handles 1D/2D arrays)
  - `predict_proba(instance: np.ndarray) -> Optional[np.ndarray]`: Get probabilities if available
  - `model_type: str` (property): Returns `"sklearn"`

**`PyTorchAdapter(ModelAdapter)`**
- Adapter for PyTorch `nn.Module` models
- **Methods**:
  - `__init__(model: Any, device: str = "cpu")`: Initialize with PyTorch model
  - `predict(instance: Any) -> int`: Get class prediction (argmax of output)
  - `predict_proba(instance: Any) -> np.ndarray`: Get class probabilities (softmax)
  - `model_type: str` (property): Returns `"pytorch"`

**`TensorFlowAdapter(ModelAdapter)`**
- Adapter for TensorFlow/Keras models
- **Methods**:
  - `__init__(model: Any)`: Initialize with TensorFlow model
  - `predict(instance: np.ndarray) -> int`: Get class prediction
  - `predict_proba(instance: np.ndarray) -> np.ndarray`: Get class probabilities
  - `model_type: str` (property): Returns `"tensorflow"`

##### Functions

**`detect_and_create_adapter(model: Any) -> ModelAdapter`**
- Auto-detect model type and create appropriate adapter
- Checks for PyTorch, scikit-learn, TensorFlow/Keras, XGBoost/LightGBM/CatBoost
- Falls back to `GenericAdapter` if no specific adapter matches

#### File: `src/adapters/languages.py`

**Purpose**: Observation languages for converting different data types to logical atoms.

##### Classes

**`TabularObservationLanguage(ObservationLanguage)`**
- Observation language for tabular/structured data
- **Attributes**:
  - `feature_names: List[str]`: Names of features
  - `label_names: List[str]`: Names of class labels
  - `discretize: bool`: Whether to discretize continuous features
  - `bins: int`: Number of bins for discretization
  - `bin_labels: List[str]`: Labels for bins (e.g., `["low", "medium", "high"]`)
- **Methods**:
  - `__init__(feature_names: List[str], label_names: List[str], discretize: bool = True, bins: int = 3, bin_labels: Optional[List[str]] = None)`: Initialize
  - `fit_discretizer(X: np.ndarray) -> None`: Fit discretizer on data (computes percentiles)
  - `discretize_value(feature_name: str, value: float) -> str`: Discretize a feature value
  - `get_predicates() -> List[str]`: Returns `["predict", "feature", "has_feature"]`
  - `generate_instance_facts(instance_id: Any, instance_data: Dict[str, Any], prediction: Any, label_name: str) -> List[Atom]`: Generate facts:
    - `predict(instance_id, label_name)`
    - `feature(instance_id, feature_name, value)` for each feature
    - `has_feature(instance_id, feature_name)` for each feature
  - `query_atom(atom: Atom, instances: Dict[Any, Any], predictions: Dict[Any, Any]) -> bool`: Check if atom is true
  - `language_name: str` (property): Returns `"tabular"`

**`ImageObservationLanguage(ObservationLanguage)`**
- Observation language for image data
- **Attributes**:
  - `label_names: List[str]`: Names of class labels
  - `image_size: tuple`: Expected image size (default: `(28, 28)`)
  - `grid_size: int`: Grid divisions for region analysis (default: `4`)
  - `intensity_levels: int`: Number of intensity levels (default: `3`)
- **Methods**:
  - `__init__(label_names: List[str], image_size: tuple = (28, 28), grid_size: int = 4, intensity_levels: int = 3)`: Initialize
  - `_analyze_image(image: np.ndarray) -> Dict[str, Any]`: Analyze image features (intensity, regions)
  - `get_predicates() -> List[str]`: Returns `["predict", "is_dark", "is_bright", "region_intensity", "has_content"]`
  - `generate_instance_facts(instance_id: Any, instance_data: np.ndarray, prediction: Any, label_name: str) -> List[Atom]`: Generate facts:
    - `predict(instance_id, label_name)`
    - `is_dark(instance_id)`, `is_bright(instance_id)`, `has_content(instance_id)` if applicable
    - `region_intensity(instance_id, region_name, level)` for each grid region
  - `query_atom(atom: Atom, instances: Dict[Any, Any], predictions: Dict[Any, Any]) -> bool`: Check if atom is true
  - `language_name: str` (property): Returns `"image"`

**`TextObservationLanguage(ObservationLanguage)`**
- Observation language for text data
- **Attributes**:
  - `label_names: List[str]`: Names of class labels
  - `vocabulary: Set[str]`: Set of words in vocabulary
  - `max_vocab_size: int`: Maximum vocabulary size (default: `1000`)
  - `min_word_freq: int`: Minimum word frequency to include (default: `2`)
- **Methods**:
  - `__init__(label_names: List[str], vocabulary: Optional[Set[str]] = None, max_vocab_size: int = 1000, min_word_freq: int = 2)`: Initialize
  - `_tokenize(text: str) -> List[str]`: Tokenize text into words
  - `update_vocabulary(tokens: List[str]) -> None`: Update vocabulary from tokens
  - `get_predicates() -> List[str]`: Returns `["predict", "has_word", "word_count_level", "is_short", "is_long"]`
  - `generate_instance_facts(instance_id: Any, instance_data: str, prediction: Any, label_name: str) -> List[Atom]`: Generate facts:
    - `predict(instance_id, label_name)`
    - `has_word(instance_id, word)` for words in vocabulary
    - `is_short(instance_id)`, `is_long(instance_id)` if applicable
    - `word_count_level(instance_id, level)` (very_short/short/medium/long)
  - `query_atom(atom: Atom, instances: Dict[Any, Any], predictions: Dict[Any, Any]) -> bool`: Check if atom is true
  - `language_name: str` (property): Returns `"text"`

#### File: `src/adapters/universal_oracle.py`

**Purpose**: Universal oracle that combines model adapter and observation language.

##### Classes

**`UniversalOracle(Oracle)`**
- Combines a model adapter and observation language to provide a unified interface
- **Attributes**:
  - `adapter: ModelAdapter`: Model adapter for predictions
  - `observation_language: ObservationLanguage`: Language for generating/querying atoms
  - `label_map: Dict[str, Any]`: Mapping of label names to model output values
  - `reverse_label_map: Dict[Any, str]`: Reverse mapping
- **Methods**:
  - `__init__(adapter: ModelAdapter, observation_language: ObservationLanguage, label_map: Dict[str, Any])`: Initialize
  - `add_instance(instance_id: Any, instance_data: Any) -> None`: Add an instance
  - `add_instances(instances: Dict[Any, Any]) -> None`: Add multiple instances
  - `get_prediction(instance_id: Any) -> Optional[str]`: Get prediction label for instance
  - `get_prediction_proba(instance_id: Any) -> Optional[np.ndarray]`: Get prediction probabilities
  - `_prepare_input(instance_data: Any) -> Any`: Prepare instance data for model input
  - `generate_facts(instance_ids: Optional[List[Any]] = None) -> List[Atom]`: Generate all facts for instances
  - `query(atom: Atom) -> bool`: Query the oracle about a ground atom (implements `Oracle` interface)
  - `get_instance_count() -> int`: Return number of instances
  - `get_instance_ids() -> List[Any]`: Return all instance IDs
  - `clear_cache() -> None`: Clear prediction cache
  - `clear_all() -> None`: Clear all instances and caches

#### File: `src/adapters/factory.py`

**Purpose**: Main API functions for easy plug-and-play explainability.

##### Classes

**`ExplainabilityResult`**
- Container for explainability results
- **Attributes**:
  - `theory: Theory`: The learned theory
  - `history: List[Theory]`: Evolution of theory during learning
  - `oracle: UniversalOracle`: The oracle used
  - `metrics: Dict[str, Any]`: Computed metrics
- **Properties**:
  - `rules: List[Clause]`: Learned rules (clauses with body)
  - `facts: List[Clause]`: Learned facts (clauses without body)
  - `num_rules: int`: Number of rules
  - `num_clauses: int`: Total number of clauses
- **Methods**:
  - `get_rules_for_label(label: str) -> List[Clause]`: Get rules that predict a specific label
  - `explain_prediction(instance_id: Any) -> Dict[str, Any]`: Explain why model made a prediction
    - Returns: `{"prediction": label, "matching_rules": [...], "key_features": [...], "num_matching_rules": int}`
  - `compute_metrics(test_instances: Optional[List[Any]] = None) -> Dict[str, float]`: Calculate metrics
    - Returns: `{"rule_coverage": float, "avg_rule_length": float, "interpretability_score": float, ...}`
  - `summary() -> str`: Generate human-readable summary
  - `__str__() -> str`: String representation (calls `summary()`)

##### Functions

**`create_explainer(model: Any, X: np.ndarray, y: Optional[np.ndarray] = None, feature_names: Optional[List[str]] = None, label_names: Optional[List[str]] = None, data_type: str = "tabular", max_instances: int = 100, verbose: bool = True, **kwargs) -> ExplainabilityResult`**
- Main entry point for creating an explainer
- **Parameters**:
  - `model`: Any trained ML model
  - `X`: Input data
  - `y`: Optional ground truth labels
  - `feature_names`: Optional feature names (auto-generated if not provided)
  - `label_names`: Optional class names (auto-detected if not provided)
  - `data_type`: `"tabular"` (default) | `"image"` | `"text"`
  - `max_instances`: Maximum instances to analyze (default: 100)
  - `verbose`: Whether to print progress (default: True)
  - `**kwargs`: Additional arguments:
    - `discretize: bool` (default: True): Discretize continuous features
    - `bins: int` (default: 3): Number of bins for discretization
    - `max_iterations: int` (default: 50): Max iterations per fact
    - `max_theory_size: int` (default: 30): Max clauses in theory
    - `min_support: float` (default: 0.1): Minimum support for rules
    - `min_confidence: float` (default: 0.6): Minimum confidence for rules
    - `image_size: tuple` (default: (28, 28)): Expected image size
    - `grid_size: int` (default: 4): Grid divisions for images
    - `max_vocab_size: int` (default: 1000): Maximum vocabulary size for text
- **Returns**: `ExplainabilityResult` object

**`explain_model(model: Any, X: np.ndarray, **kwargs) -> str`**
- One-liner convenience function
- Returns human-readable explanation string
- Calls `create_explainer()` and returns `result.summary()`

---

### Module: `src/inference/` - Model Inference Algorithm

This module implements Shapiro's Model Inference Algorithm.

#### File: `src/inference/oracle.py`

**Purpose**: Oracle interface for querying black-box models.

##### Classes

**`Oracle` (Abstract Base Class)**
- Abstract interface for oracles
- **Methods** (abstract):
  - `query(atom: Atom) -> bool`: Query about a ground atom, returns True if atom is true
- **Methods** (concrete):
  - `is_ground(atom: Atom) -> bool`: Check if atom is ground

**`NNOracle(Oracle)`**
- Legacy oracle implementation for neural networks (replaced by `UniversalOracle`)
- **Attributes**:
  - `model: Any`: The neural network model
  - `instances: Dict[int, Dict[str, Any]]`: Instance data
  - `feature_names: List[str]`: Feature names
  - `label_map: Dict[str, Any]`: Label mapping
  - `feature_value_map: Optional[Dict[str, Dict[str, Any]]]`: Feature value mapping
- **Methods**:
  - `__init__(model: Any, instances: Dict[int, Dict[str, Any]], feature_names: List[str], label_map: Dict[str, Any], feature_value_map: Optional[Dict[str, Dict[str, Any]]] = None)`: Initialize
  - `query(atom: Atom) -> bool`: Query about an atom
  - `_query_predict(atom: Atom) -> bool`: Query a `predict` atom
  - `_query_feature(atom: Atom) -> bool`: Query a `feature` atom
  - `_query_has_feature(atom: Atom) -> bool`: Query a `has_feature` atom
  - `_extract_feature_vector(instance_features: Dict[str, Any]) -> np.ndarray`: Extract feature vector
  - `add_instance(instance_id: int, features: Dict[str, Any]) -> None`: Add an instance
  - `clear_cache() -> None`: Clear prediction cache

#### File: `src/inference/algorithm.py`

**Purpose**: Main implementation of Shapiro's Algorithm 2.

##### Classes

**`ModelInference`**
- Implements Shapiro's incremental Algorithm 2 for model inference
- **Attributes**:
  - `oracle: Oracle`: The oracle for querying
  - `resolution_engine: ResolutionEngine`: Resolution engine for proofs
  - `backtracer: ContradictionBacktracer`: Backtracer for contradictions
  - `max_iterations: int`: Maximum iterations per fact
  - `max_theory_size: int`: Maximum clauses in theory
  - `min_support: float`: Minimum support for rules
  - `min_confidence: float`: Minimum confidence for rules
  - `verbose: bool`: Whether to print debug info
  - `history: List[Theory]`: Evolution of theory
  - `observed_facts: List[Atom]`: All observed facts
  - `feature_facts: List[Atom]`: Feature facts
  - `predict_facts: List[Atom]`: Prediction facts
  - `instance_features: Dict[Any, Dict[str, Any]]`: Instance feature maps
  - `instance_labels: Dict[Any, Any]`: Instance label map
  - `seen_labels: Set[Any]`: Set of seen labels
  - `feature_names: Set[str]`: Set of feature names
  - `feature_values: Dict[str, Set[Any]]`: Feature value sets
- **Methods**:
  - `__init__(oracle: Oracle, refinement_operator: Optional[RefinementOperator] = None, max_iterations: int = 100, max_theory_size: int = 50, min_support: float = 0.1, min_confidence: float = 0.6, verbose: bool = False)`: Initialize
  - `infer_theory(fact_stream: Iterator[Atom]) -> Theory`: Main inference method
    - Phase 1: Collect and index all facts
    - Phase 2: Generate pattern-based rules
    - Phase 3: Refine theory
  - `_reset_data_structures() -> None`: Reset all data structures
  - `_collect_facts(fact_stream: Iterator[Atom]) -> None`: Collect and index facts
  - `_index_feature_fact(fact: Atom) -> None`: Index a feature fact
  - `_index_predict_fact(fact: Atom) -> None`: Index a predict fact
  - `_generate_pattern_based_rules(theory: Theory) -> Theory`: Generate rules from patterns
  - `_find_rules_for_label(label: Any) -> List[Clause]`: Find rules for a label
  - `_find_single_feature_rules(label: Any, positive_instances: List[Any], negative_instances: List[Any]) -> List[Clause]`: Find single-feature rules
  - `_find_feature_pair_rules(label: Any, positive_instances: List[Any], negative_instances: List[Any]) -> List[Clause]`: Find feature-pair rules
  - `_rule_confidence(rule: Clause, positive_instances: List[Any], negative_instances: List[Any]) -> float`: Calculate rule confidence
  - `_rule_support(rule: Clause, instances: List[Any]) -> float`: Calculate rule support
  - `_rule_covers(rule: Clause, instance_id: Any) -> bool`: Check if rule covers instance
  - `_refine_theory(theory: Theory) -> Theory`: Refine theory to improve coverage
  - `_rule_label(rule: Clause) -> Optional[Any]`: Get label from rule head
  - `_create_rule_for_instance(instance_id: Any, label: Any) -> Optional[Clause]`: Create rule for instance
  - `_rank_features_for_instance(instance_id: Any, label: Any) -> List[Tuple[str, Any]]`: Rank features by discriminativeness
  - `get_history() -> List[Theory]`: Get theory evolution history

#### File: `src/inference/backtracing.py`

**Purpose**: Contradiction backtracing algorithm.

##### Classes

**`ContradictionBacktracer`**
- Implements contradiction backtracing to find false hypotheses
- **Attributes**:
  - `resolution_engine: ResolutionEngine`: Resolution engine for proofs
- **Methods**:
  - `__init__(resolution_engine: Optional[ResolutionEngine] = None)`: Initialize
  - `find_contradiction(theory: Theory, fact: Atom, oracle: Oracle) -> Optional[Clause]`: Find contradiction (theory proves fact but oracle says false), returns false clause
  - `_backtrace_contradiction(proof: ProofNode, oracle: Oracle) -> Optional[Clause]`: Backtrace through proof tree
  - `_backtrace_recursive(node: ProofNode, oracle: Oracle) -> Optional[Clause]`: Recursively backtrace
  - `find_all_contradictions(theory: Theory, facts: List[Atom], oracle: Oracle) -> List[Clause]`: Find all clauses causing contradictions
  - `is_too_strong(theory: Theory, facts: List[Atom], oracle: Oracle) -> bool`: Check if theory is too strong (proves false facts)

#### File: `src/inference/refinement.py`

**Purpose**: Refinement operators for generating new hypotheses.

##### Classes

**`RefinementOperator` (Abstract Base Class)**
- Abstract base class for refinement operators
- **Methods** (abstract):
  - `refine(clause: Clause, oracle: Oracle, observed_facts: List[Atom]) -> List[Clause]`: Generate refinements of a clause

**`AddConditionRefinement(RefinementOperator)`**
- Refinement operator that adds conditions to clause bodies
- **Attributes**:
  - `feature_names: List[str]`: Feature names to use
  - `feature_values: Dict[str, List[str]]`: Feature value mappings
- **Methods**:
  - `__init__(feature_names: List[str], feature_values: Optional[Dict[str, List[str]]] = None)`: Initialize
  - `refine(clause: Clause, oracle: Oracle, observed_facts: List[Atom]) -> List[Clause]`: Add feature conditions to body

**`SpecializePredicateRefinement(RefinementOperator)`**
- Refinement operator that specializes predicate arguments
- **Methods**:
  - `refine(clause: Clause, oracle: Oracle, observed_facts: List[Atom]) -> List[Clause]`: Specialize variable arguments to constants
  - `_get_observed_values(feature_name_arg: Atom, observed_facts: List[Atom], oracle: Oracle) -> Set[str]`: Get observed values for a feature

**`CompositeRefinementOperator(RefinementOperator)`**
- Composite operator that combines multiple operators
- **Attributes**:
  - `operators: List[RefinementOperator]`: List of operators to combine
- **Methods**:
  - `__init__(operators: List[RefinementOperator])`: Initialize
  - `refine(clause: Clause, oracle: Oracle, observed_facts: List[Atom]) -> List[Clause]`: Apply all operators and combine results

---

### Module: `src/models/` - Neural Network Models

#### File: `src/models/nn_model.py`

**Purpose**: Simple neural network classifier and data generation utilities.

##### Classes

**`SimpleNNClassifier(nn.Module)`**
- Simple feedforward neural network for binary classification
- **Attributes**:
  - `network: nn.Sequential`: The neural network layers
  - `num_classes: int`: Number of output classes
- **Methods**:
  - `__init__(input_size: int, hidden_sizes: List[int] = [64, 32], num_classes: int = 2)`: Initialize
  - `forward(x: torch.Tensor) -> torch.Tensor`: Forward pass
  - `predict(x: np.ndarray) -> int`: Predict class for input
  - `predict_proba(x: np.ndarray) -> np.ndarray`: Predict class probabilities
  - `train_model(X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001, validation_split: float = 0.2, verbose: bool = True) -> Dict[str, List[float]]`: Train the model, returns training history

##### Functions

**`generate_synthetic_loan_data(n_samples: int = 1000, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str]]`**
- Generate synthetic loan approval data
- Returns: `(X, y, feature_names)` where `X` is features, `y` is labels (0=denied, 1=approved), `feature_names` is list of feature names

---

### Module: `src/utils/` - Visualization and Utilities

#### File: `src/utils/visualization.py`

**Purpose**: Visualization utilities for rules, proof trees, and theory evolution.

##### Classes

**`ProofTreeVisualizer`**
- Visualizer for resolution proof trees
- **Attributes**:
  - `resolution_engine: ResolutionEngine`: Resolution engine
- **Methods**:
  - `__init__(resolution_engine: Optional[ResolutionEngine] = None)`: Initialize
  - `visualize_proof(theory: Theory, goal: Atom, oracle=None, save_path: Optional[str] = None, figsize: Tuple[int, int] = (14, 10)) -> Optional[ProofNode]`: Visualize proof tree
  - `_calculate_positions(root: ProofNode) -> Dict[int, Tuple[float, float]]`: Calculate node positions
  - `_get_max_depth(node: ProofNode) -> int`: Get maximum tree depth
  - `_get_width_at_depths(node: ProofNode) -> Dict[int, int]`: Get width at each depth
  - `_count_at_depth(node: ProofNode, widths: Dict[int, int]) -> None`: Count nodes at each depth
  - `_assign_positions(node: ProofNode, positions: Dict[int, Tuple[float, float]], node_id: int, x_min: float, x_max: float, max_depth: int) -> int`: Assign positions to nodes
  - `_draw_tree(ax, node: ProofNode, positions: Dict[int, Tuple[float, float]]) -> None`: Draw tree on axes
  - `_draw_node(ax, node: ProofNode, pos: Tuple[float, float]) -> None`: Draw a single node
  - `print_proof_tree(proof: ProofNode, indent: int = 0) -> str`: Generate text representation

##### Functions

**`visualize_proof_tree(theory: Theory, goal: Atom, oracle=None, save_path: Optional[str] = None, figsize: Tuple[int, int] = (14, 10)) -> Optional[ProofNode]`**
- Convenience function to visualize a proof tree

**`print_proof_tree(theory: Theory, goal: Atom, oracle=None) -> None`**
- Print text representation of proof tree

**`visualize_all_proofs(theory: Theory, facts: List[Atom], oracle=None, max_proofs: int = 5, save_dir: Optional[str] = None) -> List[ProofNode]`**
- Visualize proofs for multiple facts

**`plot_theory_evolution(history: List[Theory], save_path: Optional[str] = None) -> None`**
- Plot theory size evolution over time

**`display_rules(theory: Theory, max_rules: Optional[int] = None) -> None`**
- Display learned rules in readable format

**`plot_rule_coverage(theory: Theory, facts: List, oracle, save_path: Optional[str] = None) -> Dict[Clause, int]`**
- Plot which rules cover which facts

**`plot_rule_accuracy(theory: Theory, test_facts: List, oracle, save_path: Optional[str] = None) -> Dict[str, float]`**
- Plot accuracy metrics for learned theory

**`compare_theories(theories: List[Theory], labels: List[str], save_path: Optional[str] = None) -> None`**
- Compare multiple theories side by side

---

## API Reference

### Main Functions

#### `create_explainer(model, X, y=None, **kwargs) -> ExplainabilityResult`

Create an explainer for any model. This is the main entry point.

**Parameters:**
- `model: Any` - Any trained ML model (sklearn, PyTorch, TensorFlow, or custom with `predict()` method)
- `X: np.ndarray` - Input data (features)
- `y: Optional[np.ndarray]` - Optional ground truth labels (for validation)
- `feature_names: Optional[List[str]]` - Names for features (auto-generated if not provided)
- `label_names: Optional[List[str]]` - Names for class labels (auto-detected if not provided)
- `data_type: str` - `"tabular"` (default) | `"image"` | `"text"`
- `max_instances: int` - Maximum instances to analyze (default: 100)
- `verbose: bool` - Whether to print progress (default: True)
- `discretize: bool` - Discretize continuous features for tabular data (default: True)
- `bins: int` - Number of bins for discretization (default: 3)
- `max_iterations: int` - Max iterations per fact in algorithm (default: 50)
- `max_theory_size: int` - Maximum clauses in learned theory (default: 30)
- `min_support: float` - Minimum support for rules (default: 0.1)
- `min_confidence: float` - Minimum confidence for rules (default: 0.6)
- `image_size: tuple` - Expected image size for image data (default: (28, 28))
- `grid_size: int` - Grid divisions for images (default: 4)
- `max_vocab_size: int` - Maximum vocabulary size for text data (default: 1000)

**Returns:** `ExplainabilityResult` object with learned rules and metrics

#### `explain_model(model, X, **kwargs) -> str`

One-liner convenience function that returns a human-readable explanation string.

**Returns:** Formatted string with summary of learned rules and metrics

### ExplainabilityResult

Container for explainability results.

**Properties:**
- `.theory: Theory` - The learned Theory object (collection of clauses)
- `.rules: List[Clause]` - List of learned rules (clauses with body)
- `.facts: List[Clause]` - List of learned facts (clauses without body)
- `.num_rules: int` - Number of learned rules
- `.num_clauses: int` - Total number of clauses
- `.history: List[Theory]` - Evolution of theory during learning
- `.oracle: UniversalOracle` - The UniversalOracle used for queries
- `.metrics: Dict[str, Any]` - Dictionary of computed metrics

**Methods:**
- `.summary() -> str` - Generate human-readable summary with metrics and rules
- `.explain_prediction(instance_id) -> dict` - Explain why model made a prediction for an instance
  - Returns: `{"prediction": label, "matching_rules": [...], "key_features": [...], "num_matching_rules": int}`
- `.compute_metrics(test_instances=None) -> dict` - Calculate explainability metrics
  - Returns: `{"rule_coverage": float, "avg_rule_length": float, "interpretability_score": float, ...}`
- `.get_rules_for_label(label) -> List[Clause]` - Get all rules that predict a specific label

---

## How It Works: Step-by-Step

### 1. User Calls `create_explainer()`

```python
result = create_explainer(model, X, feature_names=["age", "income"])
```

### 2. Model Detection and Adaptation

- `detect_and_create_adapter()` automatically detects the model type:
  - Checks if it's a PyTorch `nn.Module`
  - Checks if it's from scikit-learn
  - Checks if it's from TensorFlow/Keras
  - Falls back to `GenericAdapter` for any model with `predict()` method
- Creates appropriate adapter that wraps the model with unified interface

### 3. Observation Language Setup

- Based on `data_type` parameter, creates appropriate observation language:
  - **Tabular**: `TabularObservationLanguage` - discretizes continuous features
  - **Image**: `ImageObservationLanguage` - analyzes image regions and intensity
  - **Text**: `TextObservationLanguage` - tokenizes and builds vocabulary
- For tabular data, fits discretizer on data (computes percentiles for binning)

### 4. Oracle Creation

- Creates `UniversalOracle` that combines:
  - Model adapter (for getting predictions)
  - Observation language (for generating/querying atoms)
  - Label mapping (maps label names to model output values)
- Adds all instances to oracle
- Generates facts for each instance:
  - `predict(instance_id, label_name)` - model's prediction
  - `feature(instance_id, feature_name, value)` - feature values
  - Additional facts depending on data type

### 5. Theory Learning (`ModelInference.infer_theory()`)

The algorithm runs in three phases:

**Phase 1: Fact Collection**
- Collects all facts from the oracle
- Indexes feature facts: builds `instance_features` map
- Indexes prediction facts: builds `instance_labels` map
- Extracts feature names and values

**Phase 2: Pattern-Based Rule Generation**
- For each label:
  - Finds positive instances (with this label) and negative instances
  - **Single-feature rules**: For each feature-value pair:
    - Calculates support: fraction of instances with this pattern
    - Calculates confidence: precision (positive instances with pattern / all instances with pattern)
    - If support ≥ `min_support` and confidence ≥ `min_confidence`, creates rule:
      ```
      predict(X, label) :- feature(X, feature_name, value).
      ```
  - **Feature-pair rules**: If single features aren't good enough, tries pairs:
    - Similar calculation but for combinations of two features
    - Higher confidence threshold (0.7)
- Sorts rules by confidence and support, keeps top rules

**Phase 3: Refinement**
- Finds instances not covered by any rule
- For uncovered instances, creates specific rules using most discriminative features
- Filters rules that cause too many contradictions (false positives)

### 6. Result Generation

- Creates `ExplainabilityResult` with:
  - Learned theory (collection of clauses)
  - Theory evolution history
  - Oracle (for future queries)
- Computes metrics:
  - Rule coverage: fraction of predictions explained by rules
  - Average rule length: average conditions per rule
  - Interpretability score: composite metric (coverage × length factor)

### 7. Explanation

For any instance, `explain_prediction()`:
- Gets model's prediction
- Finds all rules that predict this label
- Checks which rules' bodies are satisfied for this instance
- Extracts key features that triggered the rules
- Returns explanation with matching rules and features

---

## Examples

### Example 1: Scikit-learn Model

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from src import create_explainer

# Generate data
X, y = make_classification(n_samples=500, n_features=4, random_state=42)

# Train model
model = RandomForestClassifier().fit(X, y)

# Explain
result = create_explainer(
    model=model,
    X=X,
    feature_names=["age", "income", "credit", "history"],
    label_names=["denied", "approved"],
    max_instances=100
)

# View results
print(result.summary())

# Explain specific prediction
explanation = result.explain_prediction(0)
print(f"Prediction: {explanation['prediction']}")
print(f"Key features: {explanation['key_features']}")
```

### Example 2: PyTorch Model

```python
import torch.nn as nn
from src import create_explainer

# Define model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Train model (omitted for brevity)
model = MyModel()
# ... training code ...

# Explain
result = create_explainer(model, X_test, label_names=["class_0", "class_1"])
print(result.summary())
```

### Example 3: Image Data

```python
from src import create_explainer

# X_images is array of images (n_samples, height, width, channels)
result = create_explainer(
    model=image_classifier,
    X=X_images,
    data_type="image",
    image_size=(28, 28),
    grid_size=4
)
```

### Example 4: Text Data

```python
from src import create_explainer

# X_texts is array of strings
result = create_explainer(
    model=text_classifier,
    X=X_texts,
    data_type="text",
    max_vocab_size=1000
)
```

---

## Extending the System

### Custom Model Adapter

```python
from src.adapters import ModelAdapter
import numpy as np

class MyAdapter(ModelAdapter):
    def __init__(self, model):
        self.model = model
    
    def predict(self, instance):
        return self.model.my_predict(instance)
    
    def predict_proba(self, instance):
        return self.model.my_proba(instance) if hasattr(self.model, 'my_proba') else None
    
    @property
    def model_type(self):
        return "custom"
```

### Custom Observation Language

```python
from src.adapters import ObservationLanguage
from src.core.atoms import Atom, Constant

class MyLanguage(ObservationLanguage):
    def get_predicates(self):
        return ["predict", "my_predicate"]
    
    def generate_instance_facts(self, instance_id, instance_data, prediction, label_name):
        facts = [
            Atom("predict", [Constant(instance_id), Constant(label_name)])
        ]
        # Add custom facts based on instance_data
        # ...
        return facts
    
    def query_atom(self, atom, instances, predictions):
        # Check if atom is true for the given instances
        # ...
        return True  # or False
    
    @property
    def language_name(self):
        return "custom"
```

### Using Custom Components

```python
from src.adapters import UniversalOracle
from src.inference.algorithm import ModelInference

# Create custom adapter and language
adapter = MyAdapter(my_model)
language = MyLanguage(...)
oracle = UniversalOracle(adapter, language, label_map)

# Run inference
inference = ModelInference(oracle, max_theory_size=50)
theory = inference.infer_theory(fact_stream)
```

---

## Testing

Run the test suite:

```bash
pytest tests/
```

Or run specific test files:

```bash
pytest tests/test_atoms.py
pytest tests/test_resolution.py
pytest tests/test_backtracing.py
```

---

## Visualization

The system includes visualization utilities in `src/utils/visualization.py`:

- `plot_theory_evolution()`: Visualize how the theory evolves during learning
- `display_rules()`: Pretty-print learned rules
- `plot_rule_coverage()`: Show which instances are covered by which rules
- `plot_rule_accuracy()`: Compare rule predictions with model predictions
- `visualize_proof_tree()`: Generate proof tree visualizations for provable facts
- `print_proof_tree()`: Text-based proof tree display

Example:

```python
from src.utils.visualization import plot_theory_evolution, visualize_proof_tree

# Plot theory evolution
plot_theory_evolution(result.history, save_path="evolution.png")

# Visualize proof for a fact
from src.core.atoms import Atom, Constant
goal = Atom("predict", [Constant(0), Constant("approved")])
visualize_proof_tree(result.theory, goal, oracle=result.oracle)
```

---

## Project Structure

```
2821-final-project/
├── src/
│   ├── adapters/              # Model adapters and observation languages
│   │   ├── __init__.py        # Exports adapters, languages, factory
│   │   ├── base.py            # Abstract interfaces (ModelAdapter, ObservationLanguage)
│   │   ├── adapters.py        # Concrete adapters (SklearnAdapter, PyTorchAdapter, etc.)
│   │   ├── languages.py       # Observation languages (Tabular, Image, Text)
│   │   ├── universal_oracle.py # UniversalOracle combining adapter + language
│   │   └── factory.py         # Main API (create_explainer, explain_model, ExplainabilityResult)
│   ├── core/                  # Core logical reasoning components
│   │   ├── __init__.py        # Exports core classes
│   │   ├── atoms.py           # Atom representation, terms, variables, unification
│   │   ├── clauses.py         # Horn clause representation
│   │   ├── theory.py          # Theory (collection of clauses)
│   │   └── resolution.py      # SLD resolution engine for logical inference
│   ├── inference/             # Model inference algorithm
│   │   ├── __init__.py        # Exports inference classes
│   │   ├── algorithm.py       # ModelInference - Shapiro's Algorithm 2 implementation
│   │   ├── oracle.py          # Oracle interface (NNOracle, etc.)
│   │   ├── backtracing.py    # ContradictionBacktracer for handling contradictions
│   │   └── refinement.py      # Refinement operators for rule generalization
│   ├── models/                # Neural network models and wrappers
│   │   ├── __init__.py
│   │   └── nn_model.py        # SimpleNNClassifier and data generation utilities
│   ├── demos/                 # Demonstration scripts
│   │   ├── __init__.py
│   │   ├── plug_and_play_demo.py    # Main demo showing plug-and-play API
│   │   └── classification_demo.py   # Detailed demo with neural network classifier
│   ├── utils/                 # Visualization and utility functions
│   │   ├── __init__.py
│   │   └── visualization.py   # Plotting functions for rules, proof trees, theory evolution
│   └── __init__.py            # Main package exports
├── tests/                     # Unit tests
│   ├── __init__.py
│   ├── test_atoms.py          # Tests for atom operations
│   ├── test_resolution.py     # Tests for resolution engine
│   └── test_backtracing.py    # Tests for backtracing
├── notebooks/                 # Jupyter notebooks for experimentation
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup configuration
└── README.md                  # This file
```

---

## References

Shapiro, E. Y. (1981). An Algorithm that Infers Theories from Facts. *IJCAI*, 446-451.

---

## License

MIT
