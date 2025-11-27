"""
Demo: Plug-and-Play Explainability for Any Model

This demo shows how to use the new generalized API to explain
any machine learning model with just a few lines of code.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src import create_explainer, explain_model


def demo_sklearn_model():
    """Demo with scikit-learn model."""
    print("=" * 60)
    print("DEMO 1: Scikit-learn Model (RandomForest)")
    print("=" * 60)
    print()
    
    # Import sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=500,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Explain with one line!
    print("One-liner explanation:")
    print(explain_model(model, X, feature_names=["age", "income", "credit", "history"]))
    
    # Or get detailed results
    print("\nDetailed analysis:")
    result = create_explainer(
        model=model,
        X=X,
        feature_names=["age", "income", "credit", "history"],
        label_names=["denied", "approved"],
        max_instances=100
    )
    
    # Explain a specific prediction
    print("\nExplaining prediction for instance 0:")
    explanation = result.explain_prediction(0)
    print(f"  Prediction: {explanation['prediction']}")
    print(f"  Key features: {explanation['key_features']}")
    print(f"  Matching rules: {explanation['num_matching_rules']}")
    print()


def demo_pytorch_model():
    """Demo with PyTorch model."""
    print("=" * 60)
    print("DEMO 2: PyTorch Neural Network")
    print("=" * 60)
    print()
    
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch not installed. Skipping PyTorch demo.")
        return
    
    # Define a simple model
    class SimpleNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    # Generate data with clear patterns
    np.random.seed(42)
    X = np.random.randn(500, 4).astype(np.float32)
    # Ground truth: class 1 if f1 > 0 AND f2 > 0
    y = ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(int)
    
    # Train model
    model = SimpleNN(4, 32, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    print("Training PyTorch model...")
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
    
    model.eval()
    
    # Check accuracy
    with torch.no_grad():
        preds = torch.argmax(model(X_tensor), dim=1)
        acc = (preds == y_tensor).float().mean()
        print(f"Model accuracy: {acc:.2%}")
    
    # Explain the model - use discretization!
    result = create_explainer(
        model=model,
        X=X[:100],
        feature_names=["f1", "f2", "f3", "f4"],
        label_names=["negative", "positive"],
        max_instances=100,
        discretize=True  # Important: discretize features!
    )
    
    print(result.summary())
    
    # Show prediction explanation
    print("Example explanation for instance 5:")
    explanation = result.explain_prediction(5)
    print(f"  Prediction: {explanation['prediction']}")
    print(f"  Key features: {explanation['key_features']}")
    print(f"  Matching rules: {explanation['num_matching_rules']}")


def demo_custom_model():
    """Demo with a custom model (any object with predict method)."""
    print("=" * 60)
    print("DEMO 3: Custom Model (any predict() method)")
    print("=" * 60)
    print()
    
    # Define a simple custom model
    class MyCustomModel:
        """A simple rule-based model."""
        
        def predict(self, X):
            """Predict based on simple rules."""
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            
            predictions = []
            for row in X:
                # Rule: if feature 0 > 0 AND feature 1 > 0 -> class 1
                if row[0] > 0 and row[1] > 0:
                    predictions.append(1)
                else:
                    predictions.append(0)
            
            return predictions[0] if len(predictions) == 1 else np.array(predictions)
        
        def predict_proba(self, X):
            """Return fake probabilities."""
            pred = self.predict(X)
            if isinstance(pred, int):
                return np.array([1-pred, pred])
            return np.column_stack([1-pred, pred])
    
    # Create model and data
    model = MyCustomModel()
    np.random.seed(42)
    X = np.random.randn(200, 4)
    
    # Explain the custom model
    result = create_explainer(
        model=model,
        X=X,
        feature_names=["f1", "f2", "f3", "f4"],
        label_names=["class_0", "class_1"],
        max_instances=50
    )
    
    print(result.summary())
    
    # The learned rules should capture: f1 > 0 AND f2 > 0 -> class_1
    print("\nExpected to learn rules involving f1 and f2!")


def demo_comparison():
    """Compare explainability across different models."""
    print("=" * 60)
    print("DEMO 4: Compare Multiple Models")
    print("=" * 60)
    print()
    
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    
    # Generate data
    X, y = make_classification(
        n_samples=300,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        random_state=42
    )
    
    feature_names = ["age", "income", "score", "history"]
    label_names = ["rejected", "accepted"]
    
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42),
        "SVM": SVC(random_state=42),
    }
    
    results = {}
    for name, model in models.items():
        print(f"\nAnalyzing {name}...")
        model.fit(X, y)
        
        result = create_explainer(
            model=model,
            X=X,
            feature_names=feature_names,
            label_names=label_names,
            max_instances=50,
            verbose=False
        )
        result.compute_metrics()
        results[name] = result
    
    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"\n{'Model':<25} {'Rules':>8} {'Coverage':>10} {'Interpret.':>12}")
    print("-" * 60)
    
    for name, result in results.items():
        m = result.metrics
        print(f"{name:<25} {m['num_rules']:>8} {m['rule_coverage']*100:>9.1f}% {m['interpretability_score']:>11.3f}")
    
    print()


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("PLUG-AND-PLAY EXPLAINABILITY DEMOS")
    print("=" * 60)
    print("\nThis demo shows how easy it is to explain any ML model!")
    print()
    
    # Run demos
    demo_sklearn_model()
    print("\n")
    
    demo_pytorch_model()
    print("\n")
    
    demo_custom_model()
    print("\n")
    
    demo_comparison()
    
    print("\n" + "=" * 60)
    print("All demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

