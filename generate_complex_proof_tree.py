"""
Generate a more complex proof tree example showing multi-level reasoning.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import matplotlib.pyplot as plt
from src.core.theory import Theory
from src.core.clauses import Clause
from src.core.atoms import Atom, Constant, Variable
from src.core.resolution import ResolutionEngine
from src.utils.visualization import ProofTreeVisualizer

def create_complex_proof_tree():
    """Create a more complex proof tree with multiple levels."""
    
    print("=" * 60)
    print("Complex Proof Tree Example")
    print("=" * 60)
    
    theory = Theory()
    
    # Complex rule: Multiple conditions
    rule1 = Clause(
        head=Atom("predict", [Variable("X"), Constant("APPROVED")]),
        body=[
            Atom("feature", [Variable("X"), Constant("income"), Constant("high")]),
            Atom("feature", [Variable("X"), Constant("credit_score"), Constant("high")]),
            Atom("feature", [Variable("X"), Constant("employment_years"), Constant("high")])
        ]
    )
    theory.add_clause(rule1)
    
    # Simpler rule: Just credit score
    rule2 = Clause(
        head=Atom("predict", [Variable("X"), Constant("APPROVED")]),
        body=[
            Atom("feature", [Variable("X"), Constant("credit_score"), Constant("excellent")])
        ]
    )
    theory.add_clause(rule2)
    
    # Denial rule
    rule3 = Clause(
        head=Atom("predict", [Variable("X"), Constant("DENIED")]),
        body=[
            Atom("feature", [Variable("X"), Constant("credit_score"), Constant("low")])
        ]
    )
    theory.add_clause(rule3)
    
    # Facts for instance_0
    facts = [
        Clause(head=Atom("feature", [Constant("instance_0"), Constant("income"), Constant("high")]), body=[]),
        Clause(head=Atom("feature", [Constant("instance_0"), Constant("credit_score"), Constant("high")]), body=[]),
        Clause(head=Atom("feature", [Constant("instance_0"), Constant("employment_years"), Constant("high")]), body=[]),
    ]
    
    for fact in facts:
        theory.add_clause(fact)
    
    print("\nTheory:")
    print("-" * 60)
    for i, clause in enumerate(theory.get_clauses(), 1):
        print(f"{i}. {clause}")
    
    # Prove APPROVED
    goal = Atom("predict", [Constant("instance_0"), Constant("APPROVED")])
    
    print(f"\n\nGoal: {goal}")
    print("=" * 60)
    
    engine = ResolutionEngine(max_depth=10)
    proof = engine.prove(theory, goal, oracle=None)
    
    if proof:
        visualizer = ProofTreeVisualizer(engine)
        
        print("\nProof Tree:")
        print("-" * 60)
        print(visualizer.print_proof_tree(proof))
        
        # Visualize
        fig, ax = plt.subplots(1, 1, figsize=(18, 14))
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axis('off')
        
        positions = visualizer._calculate_positions(proof)
        visualizer._draw_tree(ax, proof, positions)
        
        ax.set_title(f"Complex Proof Tree: {goal}", fontsize=18, fontweight='bold', pad=25, color='#2C3E50')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#90EE90', edgecolor='#228B22', label='Facts', alpha=0.7),
            Patch(facecolor='#87CEEB', edgecolor='#4169E1', label='Rules', alpha=0.7),
            Patch(facecolor='#FFE4B5', edgecolor='#DAA520', label='Goals', alpha=0.7),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.95)
        
        plt.tight_layout()
        
        save_path = "figures/complex_proof_tree.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n✓ Saved: {save_path}")
        plt.close()
        
        print("\n" + "=" * 60)
        print("Explanation:")
        print("=" * 60)
        print("""
This proof tree shows a more complex example with 3 conditions:

Root: predict(instance_0, APPROVED)
  ↓
Rule 1 requires ALL of:
  ├── feature(instance_0, income, high) ✓ (Fact)
  ├── feature(instance_0, credit_score, high) ✓ (Fact)
  └── feature(instance_0, employment_years, high) ✓ (Fact)

All three conditions are proven from facts, so the prediction is APPROVED!
        """)
    else:
        print("Could not prove the goal!")

if __name__ == "__main__":
    create_complex_proof_tree()



