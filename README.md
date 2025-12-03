# Shapiro's Model Inference for Neural Network Explainability

This project implements **Ehud Shapiro's Model Inference Algorithm (1981)** as an explainability system for black-box machine learning models. The system infers logical Horn clause rules that explain model behavior by learning interpretable patterns from model predictions and input features.

## Features

- **Plug-and-Play API**: Explain any model with minimal configuration
- **Framework Agnostic**: Works with scikit-learn, PyTorch, TensorFlow, and any model with a `predict()` method
- **Multiple Data Types**: Tabular, image, and text data support
- **Interpretable Output**: Human-readable logical Horn clause rules
- **Comprehensive Metrics**: Rule coverage, interpretability scores, and prediction explanations
- **Visualization Tools**: Proof tree visualization, theory evolution plots, and rule coverage analysis

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

## Supported Models

### scikit-learn
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier().fit(X_train, y_train)
result = create_explainer(model, X_test)
```

### PyTorch
```python
import torch.nn as nn
model = MyPyTorchModel()
model.load_state_dict(torch.load("model.pt"))
result = create_explainer(model, X_test)
```

### TensorFlow/Keras
```python
from tensorflow import keras
model = keras.models.load_model("model.h5")
result = create_explainer(model, X_test)
```

### Custom Models
```python
class MyModel:
    def predict(self, X):
        # Your prediction logic
        return predictions

result = create_explainer(MyModel(), X_test)
```

## Data Types

### Tabular Data (default)
```python
result = create_explainer(
    model, X,
    feature_names=["age", "income"],
    data_type="tabular",
    discretize=True
)
```

### Image Data
```python
result = create_explainer(
    model, images,
    data_type="image",
    image_size=(28, 28),
    grid_size=4
)
```

### Text Data
```python
result = create_explainer(
    model, texts,
    data_type="text",
    max_vocab_size=1000
)
```

## Project Structure

```
2821-final-project/
├── src/
│   ├── adapters/              # Model adapters and observation languages
│   │   ├── base.py            # Abstract interfaces (ModelAdapter, ObservationLanguage)
│   │   ├── adapters.py        # Concrete adapters (SklearnAdapter, PyTorchAdapter, etc.)
│   │   ├── languages.py        # Observation languages (Tabular, Image, Text)
│   │   ├── universal_oracle.py # UniversalOracle combining adapter + language
│   │   └── factory.py         # Main API (create_explainer, explain_model, ExplainabilityResult)
│   ├── core/                  # Core logical reasoning components
│   │   ├── atoms.py           # Atom representation, terms, variables, unification
│   │   ├── clauses.py         # Horn clause representation
│   │   ├── theory.py          # Theory (collection of clauses)
│   │   └── resolution.py      # SLD resolution engine for logical inference
│   ├── inference/             # Model inference algorithm
│   │   ├── algorithm.py       # ModelInference - Shapiro's Algorithm 2 implementation
│   │   ├── oracle.py          # Oracle interface (NNOracle, etc.)
│   │   ├── backtracing.py    # ContradictionBacktracer for handling contradictions
│   │   └── refinement.py      # Refinement operators for rule generalization
│   ├── models/                # Neural network models and wrappers
│   │   ├── nn_model.py        # SimpleNNClassifier and data generation utilities
│   │   └── model_wrapper.py   # Model wrapping utilities for legacy API
│   ├── demos/                 # Demonstration scripts
│   │   ├── plug_and_play_demo.py    # Main demo showing plug-and-play API
│   │   └── classification_demo.py   # Detailed demo with neural network classifier
│   └── utils/                 # Visualization and utility functions
│       └── visualization.py   # Plotting functions for rules, proof trees, theory evolution
├── tests/                     # Unit tests
│   ├── test_atoms.py          # Tests for atom operations
│   ├── test_resolution.py     # Tests for resolution engine
│   └── test_backtracing.py    # Tests for backtracing
├── notebooks/                 # Jupyter notebooks for experimentation
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup configuration
└── README.md                  # This file
```

## How It Works

The system implements Shapiro's Model Inference Algorithm to learn interpretable rules from black-box models:

1. **Model Adaptation**: The system automatically detects and wraps your model with an appropriate adapter (sklearn, PyTorch, TensorFlow, or generic) that provides a unified prediction interface.

2. **Observation Language**: Data and predictions are converted to logical atoms:
   - **Tabular**: `predict(instance, label)`, `feature(instance, name, value)`
   - **Image**: `predict(instance, label)`, `region_intensity(instance, region, level)`
   - **Text**: `predict(instance, label)`, `has_word(instance, word)`

3. **Theory Learning**: The improved Algorithm 2 implementation:
   - **Phase 1**: Collects and indexes all observation facts (features and predictions)
   - **Phase 2**: Analyzes feature-label patterns to find discriminative rules:
     - Calculates support (fraction of instances with pattern) and confidence (precision)
     - Generates single-feature and feature-pair rules that meet thresholds
   - **Phase 3**: Refines theory to improve coverage:
     - Adds rules for uncovered instances
     - Filters rules that cause too many contradictions
   
4. **Output**: Human-readable Horn clause rules:
   ```
   predict(X, approved) :- feature(X, income, high), feature(X, credit, high).
   predict(X, denied) :- feature(X, credit, low).
   ```

5. **Explanation**: For any instance, the system can:
   - Find matching rules that explain the prediction
   - Identify key features that triggered the rules
   - Compute coverage and interpretability metrics

## Examples

### Plug-and-Play Demo
Comprehensive demo showing how to explain different types of models:
```bash
python src/demos/plug_and_play_demo.py
```
This demo includes:
- scikit-learn models (RandomForest, DecisionTree, SVM, LogisticRegression)
- PyTorch neural networks
- Custom models with `predict()` method
- Model comparison across different algorithms

### Classification Demo
Detailed demo with a neural network classifier for loan approval:
```bash
python src/demos/classification_demo.py
```
This demo shows:
- Training a neural network classifier
- Learning interpretable rules from the black-box model
- Visualization of theory evolution, rule coverage, and proof trees
- Complete workflow from data generation to rule explanation

## API Reference

### `create_explainer(model, X, y=None, **kwargs) -> ExplainabilityResult`

Create an explainer for any model. This is the main entry point for the system.

**Arguments:**
- `model`: Any trained ML model (sklearn, PyTorch, TensorFlow, or custom with `predict()` method)
- `X`: Input data (numpy array or compatible format)
- `y`: Optional ground truth labels (for validation)
- `feature_names`: Optional list of feature names (auto-generated if not provided)
- `label_names`: Optional list of class names (auto-detected if not provided)
- `data_type`: `"tabular"` (default) | `"image"` | `"text"`
- `max_instances`: Maximum instances to analyze (default: 100)
- `verbose`: Whether to print progress (default: True)
- `discretize`: Discretize continuous features for tabular data (default: True)
- `bins`: Number of bins for discretization (default: 3)
- `max_iterations`: Max iterations per fact in algorithm (default: 50)
- `max_theory_size`: Maximum clauses in learned theory (default: 30)
- `min_support`: Minimum support for rules (default: 0.1)
- `min_confidence`: Minimum confidence for rules (default: 0.6)
- `image_size`: Expected image size for image data (default: (28, 28))
- `grid_size`: Grid divisions for images (default: 4)
- `max_vocab_size`: Maximum vocabulary size for text data (default: 1000)

**Returns:** `ExplainabilityResult` object with learned rules and metrics

### `explain_model(model, X, **kwargs) -> str`

One-liner convenience function that returns a human-readable explanation string.

**Returns:** Formatted string with summary of learned rules and metrics

### `ExplainabilityResult`

Container for explainability results with the following interface:

**Properties:**
- `.theory`: The learned Theory object (collection of clauses)
- `.rules`: List of learned rules (clauses with body)
- `.facts`: List of learned facts (clauses without body)
- `.num_rules`: Number of learned rules
- `.num_clauses`: Total number of clauses
- `.history`: Evolution of theory during learning
- `.oracle`: The UniversalOracle used for queries
- `.metrics`: Dictionary of computed metrics

**Methods:**
- `.summary() -> str`: Generate human-readable summary with metrics and rules
- `.explain_prediction(instance_id) -> dict`: Explain why model made a prediction for an instance
  - Returns: `{"prediction": label, "matching_rules": [...], "key_features": [...], "num_matching_rules": int}`
- `.compute_metrics(test_instances=None) -> dict`: Calculate explainability metrics
  - Returns: `{"rule_coverage": float, "avg_rule_length": float, "interpretability_score": float, ...}`
- `.get_rules_for_label(label) -> List[Clause]`: Get all rules that predict a specific label

## Core Components

### Model Adapters (`src/adapters/`)
- **ModelAdapter**: Abstract base class for wrapping ML models
- **SklearnAdapter**: Adapter for scikit-learn models
- **PyTorchAdapter**: Adapter for PyTorch `nn.Module` models
- **TensorFlowAdapter**: Adapter for TensorFlow/Keras models
- **GenericAdapter**: Fallback adapter for any model with `predict()` method
- **detect_and_create_adapter()**: Auto-detects model type and creates appropriate adapter

### Observation Languages (`src/adapters/languages.py`)
- **TabularObservationLanguage**: Converts tabular data to `feature(instance, name, value)` atoms
- **ImageObservationLanguage**: Converts images to `region_intensity(instance, region, level)` atoms
- **TextObservationLanguage**: Converts text to `has_word(instance, word)` atoms

### Core Logic (`src/core/`)
- **Atom**: Represents logical atoms with predicates and arguments
- **Clause**: Represents Horn clauses (head :- body)
- **Theory**: Collection of clauses with operations for querying and management
- **ResolutionEngine**: SLD resolution for logical inference and proof generation

### Inference Algorithm (`src/inference/`)
- **ModelInference**: Main implementation of Shapiro's Algorithm 2
  - Pattern-based rule generation from feature-label correlations
  - Support and confidence-based rule filtering
  - Theory refinement for improved coverage
- **Oracle**: Interface for querying model predictions and data
- **ContradictionBacktracer**: Handles contradictions during theory learning
- **RefinementOperator**: Operators for generalizing and specializing rules

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

## Visualization

The system includes visualization utilities in `src/utils/visualization.py`:

- `plot_theory_evolution()`: Visualize how the theory evolves during learning
- `display_rules()`: Pretty-print learned rules
- `plot_rule_coverage()`: Show which instances are covered by which rules
- `plot_rule_accuracy()`: Compare rule predictions with model predictions
- `visualize_proof_tree()`: Generate proof tree visualizations for provable facts
- `print_proof_tree()`: Text-based proof tree display

## References

Shapiro, E. Y. (1981). An Algorithm that Infers Theories from Facts. *IJCAI*, 446-451.

## License

MIT
