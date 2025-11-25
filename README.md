# Shapiro's Model Inference for Neural Network Explainability

This project implements **Ehud Shapiro's Model Inference Algorithm (1981)** as an explainability system for black-box neural network models. The system infers logical Horn clause rules that explain model behavior by querying the model and using contradiction backtracing and refinement operators.

## Overview

Given a black-box neural network (e.g., feedforward classifier), this system:
1. Queries the model on input instances
2. Observes input features and model predictions
3. Infers logical rules (Horn clauses) that explain the model's behavior
4. Refines rules through contradiction backtracing and refinement operators

## Project Structure

```
2821-final-project/
├── src/
│   ├── core/           # Core logical reasoning (atoms, clauses, resolution)
│   ├── inference/      # Model inference engine (oracle, backtracing, refinement)
│   ├── models/         # Neural network models and wrappers
│   ├── demos/          # Demo scripts
│   └── utils/          # Visualization utilities
├── tests/              # Unit tests
└── notebooks/          # Jupyter notebooks
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.models.nn_model import SimpleNNClassifier
from src.models.model_wrapper import wrap_nn
from src.inference.algorithm import ModelInference

# Train or load a neural network
model = SimpleNNClassifier(...)
model.train(...)

# Wrap model as oracle
oracle = wrap_nn(model, feature_names, label_map)

# Run model inference
inference = ModelInference(oracle)
theory = inference.infer_theory(fact_stream)

# Display learned rules
from src.utils.visualization import display_rules
display_rules(theory)
```

## Example: Loan Approval Classifier

See `src/demos/classification_demo.py` for a complete example that:
- Trains a simple feedforward NN on loan approval data
- Learns logical rules explaining the model's decisions
- Visualizes rule evolution over time

## Key Components

- **Core**: Atom representation, Horn clauses, SLD resolution engine
- **Inference**: Oracle interface, contradiction backtracing, refinement operators
- **Models**: Simple feedforward NN classifier with tabular data
- **Algorithm**: Incremental Algorithm 2 from Shapiro (1981)

## References

Shapiro, E. Y. (1981). An Algorithm that Infers Theories from Facts. *IJCAI*, 446-451.


MODEL PROGRESSION
Reviewing the project to understand the current setup and suggest a model progression.


[4 tools called]


Checking the inference algorithm and model wrapper to understand the current capabilities:


[3 tools called]


Progression of models and use cases for your Shapiro Model Inference explainability system:

## Model Progression (Increasing Complexity)

### **Level 1: Simple Feedforward Neural Networks** (Current)
**Architecture**: Basic MLP with 2-3 hidden layers  
**Data/Use Cases**:
- **Tabular classification**: Loan approval, credit scoring, medical diagnosis (binary/multiclass)
- **Simple feature interactions**: Income + credit score → approval
- **Low-dimensional data**: 4-10 features, clear decision boundaries

**Why start here**: Simple decision boundaries, easy to interpret, good baseline

---

### **Level 2: Deeper Feedforward Networks**
**Architecture**: 4-6 hidden layers, batch normalization, dropout  
**Data/Use Cases**:
- **Higher-dimensional tabular**: 20-50 features (e.g., customer churn, fraud detection)
- **Complex feature interactions**: Multiple interacting factors
- **Noisy data**: Real-world datasets with missing values, outliers

**Challenges**: Deeper networks learn more complex patterns; rules may be longer

---

### **Level 3: Convolutional Neural Networks (CNNs)**
**Architecture**: Conv layers, pooling, flatten → dense layers  
**Data/Use Cases**:
- **Image classification (small)**: MNIST, Fashion-MNIST, CIFAR-10
- **Medical imaging**: X-ray classification, skin lesion detection
- **Document classification**: Handwritten digits, simple OCR

**Adaptation needed**: Convert image pixels to logical predicates (e.g., `pixel(x, y, value)`, `region(x, y, w, h, pattern)`)

---

### **Level 4: Recurrent Neural Networks (RNNs/LSTMs)**
**Architecture**: LSTM/GRU cells for sequence processing  
**Data/Use Cases**:
- **Time series classification**: ECG signals, sensor data, stock price patterns
- **Text classification (simple)**: Sentiment analysis, spam detection
- **Sequence prediction**: Next event prediction

**Adaptation needed**: Temporal predicates (e.g., `at_time(t, feature, value)`, `precedes(event1, event2)`)

---

### **Level 5: Transformer-based Models (Simplified)**
**Architecture**: Attention mechanisms, encoder-only or small transformer  
**Data/Use Cases**:
- **Text classification**: News categorization, intent detection
- **Structured text**: Legal document classification, code analysis
- **Multi-modal (simple)**: Text + metadata classification

**Adaptation needed**: Attention patterns as predicates (e.g., `attends_to(token1, token2, weight)`)

---

### **Level 6: Graph Neural Networks (GNNs)**
**Architecture**: Graph convolution, message passing  
**Data/Use Cases**:
- **Social networks**: Community detection, node classification
- **Molecular data**: Drug discovery, chemical property prediction
- **Knowledge graphs**: Entity classification, relation prediction

**Adaptation needed**: Graph predicates (e.g., `connected(node1, node2)`, `has_feature(node, feature)`)

---

### **Level 7: Ensemble Models**
**Architecture**: Multiple models (NNs + tree-based)  
**Data/Use Cases**:
- **Hybrid systems**: Neural network + random forest ensemble
- **Multi-model explanations**: Explaining ensemble decisions
- **Production systems**: Real-world deployed models

**Adaptation needed**: Meta-predicates combining multiple model outputs

---

## Within Each Model: Data/Use Case Progression

### **Phase A: Synthetic/Controlled Data**
- Purpose: Validate the explainability system
- Characteristics: Known ground-truth rules, clean data, controlled complexity
- Example: Loan approval with explicit rules (income > X AND credit > Y)

### **Phase B: Standard Benchmarks**
- Purpose: Compare against known baselines
- Characteristics: Public datasets, established metrics
- Examples:
  - **Tabular**: UCI datasets (Adult, Wine, Iris)
  - **Images**: MNIST, CIFAR-10
  - **Text**: IMDB sentiment, 20 Newsgroups

### **Phase C: Real-World Applications**
- Purpose: Test practical utility
- Characteristics: Noisy, imbalanced, domain-specific
- Examples:
  - **Healthcare**: Medical diagnosis, drug interaction
  - **Finance**: Fraud detection, risk assessment
  - **Legal**: Document classification, case prediction

### **Phase D: Edge Cases & Challenges**
- Purpose: Stress test the system
- Characteristics: Adversarial examples, distribution shift, rare patterns
- Examples:
  - **Adversarial robustness**: Explaining misclassifications
  - **Out-of-distribution**: Explaining failures on new data
  - **Rare classes**: Explaining minority class predictions

---

## Recommended Learning Path

1. **Start**: Simple NN + synthetic loan data (current)
2. **Next**: Deeper NN + UCI Adult dataset (income prediction)
3. **Then**: CNN + MNIST (digit classification)
4. **Advanced**: LSTM + IMDB sentiment (text classification)
5. **Research**: GNN + molecular data (if time permits)

Each step should:
- Validate that learned rules match expected patterns
- Measure rule accuracy vs. model accuracy
- Analyze rule complexity vs. model complexity
- Document what patterns the system can/cannot capture

Should I help implement support for any of these model types or create demo datasets for a specific level?
