"""
Generate an example proof tree showing how the framework proves predictions.

This script creates a realistic example showing how a prediction is proven
using learned rules and feature facts.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

from src.core.theory import Theory
from src.core.clauses import Clause
from src.core.atoms import Atom, Constant, Variable
from src.core.resolution import ResolutionEngine
from src.utils.visualization import visualize_proof_tree, ProofTreeVisualizer

def create_example_proof_tree():
    """Create an example proof tree showing how predictions are proven."""
    
    print("Creating example proof tree...")
    print("=" * 60)
    
    # Create a theory with learned rules
    theory = Theory()
    
    # Rule 1: High income AND high credit -> APPROVED
    rule1 = Clause(
        head=Atom("predict", [Variable("X"), Constant("APPROVED")]),
        body=[
            Atom("feature", [Variable("X"), Constant("income"), Constant("high")]),
            Atom("feature", [Variable("X"), Constant("credit_score"), Constant("high")])
        ]
    )
    theory.add_clause(rule1)
    
    # Rule 2: Low credit -> DENIED
    rule2 = Clause(
        head=Atom("predict", [Variable("X"), Constant("DENIED")]),
        body=[
            Atom("feature", [Variable("X"), Constant("credit_score"), Constant("low")])
        ]
    )
    theory.add_clause(rule2)
    
    # Add feature facts for instance_0
    fact1 = Clause(
        head=Atom("feature", [Constant("instance_0"), Constant("income"), Constant("high")]),
        body=[]
    )
    theory.add_clause(fact1)
    
    fact2 = Clause(
        head=Atom("feature", [Constant("instance_0"), Constant("credit_score"), Constant("high")]),
        body=[]
    )
    theory.add_clause(fact2)
    
    fact3 = Clause(
        head=Atom("feature", [Constant("instance_0"), Constant("age"), Constant("medium")]),
        body=[]
    )
    theory.add_clause(fact3)
    
    print("\nTheory (learned rules and facts):")
    print("-" * 60)
    for i, clause in enumerate(theory.get_clauses(), 1):
        print(f"{i}. {clause}")
    
    # Goal: Prove predict(instance_0, APPROVED)
    goal = Atom("predict", [Constant("instance_0"), Constant("APPROVED")])
    
    print(f"\n\nGoal to prove: {goal}")
    print("=" * 60)
    
    # Create resolution engine
    engine = ResolutionEngine(max_depth=10)
    
    # Prove the goal
    proof = engine.prove(theory, goal, oracle=None)
    
    if proof is None:
        print("Could not prove the goal!")
        return None
    
    print("\n✓ Proof found!")
    print("\nProof Tree Structure:")
    print("-" * 60)
    
    # Print text representation
    visualizer = ProofTreeVisualizer(engine)
    print(visualizer.print_proof_tree(proof))
    
    # Create visualization
    print("\n\nGenerating proof tree visualization...")
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis('off')
    
    # Calculate positions
    positions = visualizer._calculate_positions(proof)
    
    # Draw the tree
    visualizer._draw_tree(ax, proof, positions)
    
    ax.set_title(f"Proof Tree: {goal}", fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#90EE90', edgecolor='#228B22', label='Facts (Ground Truth)', alpha=0.7),
        Patch(facecolor='#87CEEB', edgecolor='#4169E1', label='Rules', alpha=0.7),
        Patch(facecolor='#FFE4B5', edgecolor='#DAA520', label='Goals', alpha=0.7),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    save_path = "figures/example_proof_tree.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    
    plt.close()
    
    print("\n" + "=" * 60)
    print("Proof Tree Explanation:")
    print("=" * 60)
    print("""
The proof tree shows how the framework proves: predict(instance_0, APPROVED)

Root (Goal):
  predict(instance_0, APPROVED)
    ↓
  Uses Rule 1: predict(X, APPROVED) :- feature(X, income, high), feature(X, credit_score, high)
    ↓
  Subgoals (must prove both):
    ├── feature(instance_0, income, high)  ✓ (Fact)
    └── feature(instance_0, credit_score, high)  ✓ (Fact)

Both subgoals are proven directly from facts in the theory, so the proof succeeds!
    """)
    
    return proof


def create_multiple_proof_examples():
    """Create multiple proof tree examples."""
    
    print("\n" + "=" * 60)
    print("Creating Multiple Proof Tree Examples")
    print("=" * 60)
    
    # Example 1: Simple approval proof
    print("\nExample 1: Proving APPROVED")
    proof1 = create_example_proof_tree()
    
    # Example 2: Denial proof
    print("\n\n" + "=" * 60)
    print("Example 2: Proving DENIED")
    print("=" * 60)
    
    theory2 = Theory()
    
    # Rule: Low credit -> DENIED
    rule = Clause(
        head=Atom("predict", [Variable("X"), Constant("DENIED")]),
        body=[Atom("feature", [Variable("X"), Constant("credit_score"), Constant("low")])]
    )
    theory2.add_clause(rule)
    
    # Fact: instance_1 has low credit
    fact = Clause(
        head=Atom("feature", [Constant("instance_1"), Constant("credit_score"), Constant("low")]),
        body=[]
    )
    theory2.add_clause(fact)
    
    goal2 = Atom("predict", [Constant("instance_1"), Constant("DENIED")])
    
    engine = ResolutionEngine(max_depth=10)
    proof2 = engine.prove(theory2, goal2, oracle=None)
    
    if proof2:
        visualizer = ProofTreeVisualizer(engine)
        print("\nProof Tree:")
        print(visualizer.print_proof_tree(proof2))
        
        # Visualize
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axis('off')
        
        positions = visualizer._calculate_positions(proof2)
        visualizer._draw_tree(ax, proof2, positions)
        ax.set_title(f"Proof Tree: {goal2}", fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        save_path = "figures/example_proof_tree_denied.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n✓ Saved: {save_path}")
        plt.close()


if __name__ == "__main__":
    create_example_proof_tree()
    print("\n\nTo see more examples, run:")
    print("  python generate_proof_tree_example.py")



