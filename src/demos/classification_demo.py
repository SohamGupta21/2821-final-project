"""Demo: Learn rules explaining a neural network classifier."""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.nn_model import SimpleNNClassifier, generate_synthetic_loan_data
from src.inference.algorithm import ModelInference
from src.inference.oracle import NNOracle
from src.core.atoms import Atom, Constant
from src.core.resolution import ResolutionEngine
from src.utils.visualization import (
    plot_theory_evolution,
    display_rules,
    plot_rule_coverage,
    plot_rule_accuracy,
    visualize_proof_tree,
    print_proof_tree,
    ProofTreeVisualizer
)
from typing import Dict, List, Any, Optional


# Helper functions for legacy demo workflow
def _create_instances_from_data(
    X: np.ndarray,
    feature_names: List[str],
    discretize: bool = True,
    bins: int = 3
) -> Dict[int, Dict[str, Any]]:
    """Create instance dictionaries from feature matrix."""
    instances = {}
    n_samples = X.shape[0]
    
    for i in range(n_samples):
        instance_features = {}
        for j, feature_name in enumerate(feature_names):
            value = X[i, j]
            if discretize:
                if bins == 3:
                    if value > 0.3:
                        discrete_value = "high"
                    elif value < -0.3:
                        discrete_value = "low"
                    else:
                        discrete_value = "medium"
                else:
                    bin_edges = np.linspace(X[:, j].min(), X[:, j].max(), bins + 1)
                    bin_idx = np.digitize(value, bin_edges) - 1
                    discrete_value = f"bin_{bin_idx}"
                instance_features[feature_name] = discrete_value
            else:
                instance_features[feature_name] = float(value)
        instances[i] = instance_features
    return instances


def _wrap_nn(
    model: SimpleNNClassifier,
    instances: Dict[int, Dict[str, Any]],
    feature_names: List[str],
    label_map: Dict[str, int],
    feature_value_map: Optional[Dict[str, Dict[str, float]]] = None
) -> NNOracle:
    """Wrap a neural network model as an oracle."""
    class ModelWrapper:
        def __init__(self, nn_model):
            self.nn_model = nn_model
        def predict(self, feature_vector: np.ndarray) -> int:
            return self.nn_model.predict(feature_vector)
    
    return NNOracle(
        model=ModelWrapper(model),
        instances=instances,
        feature_names=feature_names,
        label_map=label_map,
        feature_value_map=feature_value_map
    )


def _generate_facts_from_instances(
    instances: Dict[int, Dict[str, Any]],
    model: SimpleNNClassifier,
    feature_names: List[str],
    label_map: Dict[str, int],
    instance_ids: Optional[List[int]] = None
) -> List[Atom]:
    """Generate observation facts from instances."""
    facts = []
    if instance_ids is None:
        instance_ids = list(instances.keys())
    
    for instance_id in instance_ids:
        if instance_id not in instances:
            continue
        
        instance_features = instances[instance_id]
        feature_vector = []
        for feature_name in feature_names:
            value = instance_features.get(feature_name, 0.0)
            if isinstance(value, str):
                value = float(hash(value) % 100) / 100.0
            feature_vector.append(float(value))
        
        feature_vector = np.array(feature_vector).reshape(1, -1)
        prediction = model.predict(feature_vector)
        
        label_name = None
        for name, value in label_map.items():
            if value == prediction:
                label_name = name
                break
        
        if label_name is None:
            continue
        
        facts.append(Atom("predict", [Constant(instance_id), Constant(label_name)]))
        
        for feature_name, value in instance_features.items():
            if isinstance(value, str):
                facts.append(Atom("feature", [
                    Constant(instance_id), Constant(feature_name), Constant(value)
                ]))
            else:
                facts.append(Atom("feature", [
                    Constant(instance_id), Constant(feature_name), Constant(float(value))
                ]))
    
    return facts


def main():
    """Run the classification demo."""
    print("="*60)
    print("Shapiro's Model Inference Demo")
    print("Learning Rules for Loan Approval Classifier")
    print("="*60)
    print()
    
    # Step 1: Generate synthetic data
    print("Step 1: Generating synthetic loan approval data...")
    X, y, feature_names = generate_synthetic_loan_data(n_samples=500, random_state=42)
    print(f"  Generated {len(X)} samples with {len(feature_names)} features")
    print(f"  Features: {feature_names}")
    print(f"  Class distribution: {np.bincount(y)} (DENIED, APPROVED)")
    print()
    
    # Step 2: Train neural network
    print("Step 2: Training neural network classifier...")
    model = SimpleNNClassifier(input_size=len(feature_names), hidden_sizes=[32, 16])
    train_history = model.train_model(X, y, epochs=50, verbose=True)
    print(f"  Final validation accuracy: {train_history['val_acc'][-1]:.4f}")
    print()
    
    # Step 3: Create instances and oracle
    print("Step 3: Setting up oracle...")
    instances = _create_instances_from_data(X, feature_names, discretize=True)
    label_map = {"APPROVED": 1, "DENIED": 0}
    oracle = _wrap_nn(model, instances, feature_names, label_map)
    print(f"  Created {len(instances)} instances")
    print()
    
    # Step 4: Generate observation facts (BOTH feature and predict facts)
    print("Step 4: Generating observation facts...")
    # Use a stratified sample to ensure both labels are represented
    np.random.seed(42)
    
    # Get predictions for all instances to stratify
    all_predictions = []
    for i in range(len(X)):
        pred = model.predict(X[i:i+1])
        all_predictions.append(pred)
    
    # Find indices for each class
    approved_indices = [i for i, p in enumerate(all_predictions) if p == 1]
    denied_indices = [i for i, p in enumerate(all_predictions) if p == 0]
    
    print(f"  Model predictions: {len(denied_indices)} DENIED, {len(approved_indices)} APPROVED")
    
    # Sample from each class
    n_per_class = 50
    sampled_approved = np.random.choice(approved_indices, min(n_per_class, len(approved_indices)), replace=False).tolist() if approved_indices else []
    sampled_denied = np.random.choice(denied_indices, min(n_per_class, len(denied_indices)), replace=False).tolist() if denied_indices else []
    
    train_indices = sampled_approved + sampled_denied
    np.random.shuffle(train_indices)
    
    facts = _generate_facts_from_instances(
        instances, model, feature_names, label_map, instance_ids=train_indices
    )
    
    # Count facts by type
    feature_facts = [f for f in facts if f.predicate == "feature"]
    predict_facts = [f for f in facts if f.predicate == "predict"]
    
    print(f"  Generated {len(facts)} total facts:")
    print(f"    - {len(feature_facts)} feature facts")
    print(f"    - {len(predict_facts)} prediction facts")
    
    # Show label distribution in predictions
    approved_count = sum(1 for f in predict_facts 
                        if len(f.arguments) >= 2 and 
                        isinstance(f.arguments[1], Constant) and 
                        f.arguments[1].value == "APPROVED")
    denied_count = len(predict_facts) - approved_count
    print(f"  Prediction distribution: {denied_count} DENIED, {approved_count} APPROVED")
    print()
    
    # Step 5: Run model inference
    print("Step 5: Running Model Inference Algorithm...")
    print("  (This may take a few moments...)")
    print()
    
    # Pass ALL facts (features + predictions) for learning
    inference = ModelInference(oracle, max_iterations=20, max_theory_size=20, verbose=True)
    theory = inference.infer_theory(iter(facts))
    
    print()
    print(f"  Learned theory with {len(theory)} clauses")
    print()
    
    # Step 6: Display results
    print("Step 6: Displaying learned rules...")
    display_rules(theory)
    
    # Step 7: Visualizations
    print("Step 7: Generating visualizations...")
    
    # Theory evolution
    history = inference.get_history()
    plot_theory_evolution(history)
    
    # Rule coverage (on a subset)
    test_facts = predict_facts[:20]
    plot_rule_coverage(theory, test_facts, oracle)
    
    # Rule accuracy
    plot_rule_accuracy(theory, test_facts, oracle)
    
    # Step 8: Proof Tree Visualization
    print()
    print("Step 8: Proof Tree Visualizations...")
    print()
    
    # To prove facts, we need to add feature facts to the theory as ground facts
    # OR rely on the oracle to confirm them
    engine = ResolutionEngine(max_depth=10)
    proven_facts = []
    
    # First, let's test what the theory can prove with oracle support
    print("  Testing provability with oracle...")
    
    for fact in predict_facts[:30]:
        # Try to prove using the oracle for feature checks
        can_prove = engine.can_prove(theory, fact, oracle=oracle.query)
        if can_prove:
            proven_facts.append(fact)
            print(f"    ✓ Can prove: {fact}")
        if len(proven_facts) >= 5:
            break
    
    if not proven_facts:
        # If no facts provable with oracle, create a theory with feature facts for visualization
        print("  No facts provable with oracle alone.")
        print("  Creating extended theory with feature facts for demonstration...")
        
        # Create a theory that includes feature facts as clauses
        extended_theory = theory.copy()
        
        # Add some feature facts as ground clauses
        for feat_fact in feature_facts[:50]:
            from src.core.clauses import Clause
            extended_theory.add_clause(Clause(feat_fact, []))
        
        # Now try to prove with the extended theory
        for fact in predict_facts[:30]:
            can_prove = engine.can_prove(extended_theory, fact, oracle=None)
            if can_prove:
                proven_facts.append((fact, extended_theory))
                print(f"    ✓ Can prove with extended theory: {fact}")
            if len(proven_facts) >= 3:
                break
    
    if proven_facts:
        print(f"\n  Visualizing proof trees for {len(proven_facts)} provable facts...")
        
        for i, item in enumerate(proven_facts):
            if isinstance(item, tuple):
                fact, use_theory = item
            else:
                fact = item
                use_theory = theory
            
            print(f"\n  Proof {i+1}: {fact}")
            print("-" * 50)
            
            # Print text version of proof tree
            print_proof_tree(use_theory, fact, oracle if use_theory == theory else None)
            
            # Visual proof tree (matplotlib)
            visualize_proof_tree(use_theory, fact, oracle if use_theory == theory else None, figsize=(12, 8))
    else:
        print("  No provable facts found to visualize.")
    
    print()
    print("="*60)
    print("Demo complete!")
    print("="*60)


if __name__ == "__main__":
    main()


