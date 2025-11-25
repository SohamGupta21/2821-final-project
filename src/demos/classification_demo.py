"""Demo: Learn rules explaining a neural network classifier."""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.nn_model import SimpleNNClassifier, generate_synthetic_loan_data
from src.models.model_wrapper import (
    wrap_nn,
    create_instances_from_data,
    generate_facts_from_instances
)
from src.inference.algorithm import ModelInference
from src.core.atoms import Atom, Constant
from src.utils.visualization import (
    plot_theory_evolution,
    display_rules,
    plot_rule_coverage,
    plot_rule_accuracy
)


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
    print(f"  Class distribution: {np.bincount(y)}")
    print()
    
    # Step 2: Train neural network
    print("Step 2: Training neural network classifier...")
    model = SimpleNNClassifier(input_size=len(feature_names), hidden_sizes=[32, 16])
    history = model.train_model(X, y, epochs=50, verbose=True)
    print(f"  Final validation accuracy: {history['val_acc'][-1]:.4f}")
    print()
    
    # Step 3: Create instances and oracle
    print("Step 3: Setting up oracle...")
    instances = create_instances_from_data(X, feature_names, discretize=True)
    label_map = {"APPROVED": 1, "DENIED": 0}
    oracle = wrap_nn(model, instances, feature_names, label_map)
    print(f"  Created {len(instances)} instances")
    print()
    
    # Step 4: Generate observation facts
    print("Step 4: Generating observation facts...")
    # Use a subset for faster demo
    train_indices = list(range(100))  # Use first 100 instances
    facts = generate_facts_from_instances(
        instances, model, feature_names, label_map, instance_ids=train_indices
    )
    print(f"  Generated {len(facts)} facts")
    
    # Filter to only predict facts for the main learning
    predict_facts = [f for f in facts if f.predicate == "predict"]
    print(f"  Using {len(predict_facts)} prediction facts for learning")
    print()
    
    # Step 5: Run model inference
    print("Step 5: Running Model Inference Algorithm...")
    print("  (This may take a few moments...)")
    print()
    
    inference = ModelInference(oracle, max_iterations=20, max_theory_size=20)
    theory = inference.infer_theory(iter(predict_facts))
    
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
    
    print()
    print("="*60)
    print("Demo complete!")
    print("="*60)


if __name__ == "__main__":
    main()

