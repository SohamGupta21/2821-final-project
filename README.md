# Shapiro's Model Inference for Neural Network Explainability

This project implements **Ehud Shapiro's Model Inference Algorithm (1981)** as an explainability system for black-box machine learning models. The system infers logical Horn clause rules that explain model behavior.

## Features

- **Plug-and-Play**: Explain any model with just 2 lines of code
- **Framework Agnostic**: Works with scikit-learn, PyTorch, TensorFlow, and custom models
- **Data Type Support**: Tabular, image, and text data
- **Interpretable Output**: Human-readable logical rules
- **Metrics**: Rule coverage, interpretability scores

## Installation

```bash
pip install -r requirements.txt
```

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
│   ├── adapters/       # Plug-and-play model adapters
│   │   ├── base.py           # Abstract interfaces
│   │   ├── adapters.py       # Model adapters (sklearn, pytorch, etc.)
│   │   ├── languages.py      # Observation languages (tabular, image, text)
│   │   ├── universal_oracle.py
│   │   └── factory.py        # create_explainer, ExplainabilityResult
│   ├── core/           # Core logical reasoning
│   │   ├── atoms.py          # Atom representation, unification
│   │   ├── clauses.py        # Horn clauses
│   │   ├── theory.py         # Theory (collection of clauses)
│   │   └── resolution.py     # SLD resolution engine
│   ├── inference/      # Model inference engine
│   │   ├── algorithm.py      # Shapiro's Algorithm 2
│   │   ├── oracle.py         # Oracle interface
│   │   ├── backtracing.py    # Contradiction backtracing
│   │   └── refinement.py     # Refinement operators
│   ├── models/         # Neural network models
│   ├── demos/          # Demo scripts
│   └── utils/          # Visualization utilities
├── tests/              # Unit tests
└── notebooks/          # Jupyter notebooks
```

## How It Works

1. **Model Adaptation**: The system wraps your model with an adapter that provides a unified prediction interface.

2. **Observation Language**: Data is converted to logical atoms:
   - `predict(instance, label)` - model predictions
   - `feature(instance, name, value)` - feature values
   - `has_word(instance, word)` - text tokens (for text data)
   - `region_intensity(instance, region, level)` - image regions

3. **Theory Learning**: Shapiro's algorithm iteratively:
   - Removes contradictions (theory proves false facts)
   - Adds refinements (theory can't prove true facts)
   
4. **Output**: Human-readable logical rules:
   ```
   predict(X, approved) :- feature(X, income, high), feature(X, credit, high).
   predict(X, denied) :- feature(X, credit, low).
   ```

## Examples

### Full Demo
```bash
python src/demos/plug_and_play_demo.py
```

### Loan Approval Demo
```bash
python src/demos/classification_demo.py
```

## API Reference

### `create_explainer(model, X, **kwargs)`
Create an explainer for any model.

**Arguments:**
- `model`: Any trained ML model
- `X`: Input data (numpy array)
- `feature_names`: Optional list of feature names
- `label_names`: Optional list of class names
- `data_type`: "tabular" | "image" | "text"
- `max_instances`: Maximum instances to analyze (default: 100)
- `discretize`: Discretize continuous features (default: True)

**Returns:** `ExplainabilityResult`

### `explain_model(model, X, **kwargs)`
One-liner to get explanation string.

### `ExplainabilityResult`
- `.rules`: List of learned rules
- `.summary()`: Human-readable summary
- `.explain_prediction(id)`: Explain specific instance
- `.compute_metrics()`: Calculate coverage, interpretability

## Extending the System

### Custom Model Adapter
```python
from src.adapters import ModelAdapter

class MyAdapter(ModelAdapter):
    def predict(self, instance):
        return self.model.my_predict(instance)
    
    def predict_proba(self, instance):
        return self.model.my_proba(instance)
    
    @property
    def model_type(self):
        return "custom"
```

### Custom Observation Language
```python
from src.adapters import ObservationLanguage

class MyLanguage(ObservationLanguage):
    def get_predicates(self):
        return ["predict", "my_predicate"]
    
    def generate_instance_facts(self, instance_id, data, prediction, label):
        # Convert data to atoms
        return [...]
    
    def query_atom(self, atom, instances, predictions):
        # Check if atom is true
        return True/False
```

## References

Shapiro, E. Y. (1981). An Algorithm that Infers Theories from Facts. *IJCAI*, 446-451.

## License

MIT
