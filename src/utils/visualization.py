"""Visualization utilities for rule evolution and display."""

import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from ..core.theory import Theory
from ..core.clauses import Clause


def plot_theory_evolution(history: List[Theory], save_path: Optional[str] = None) -> None:
    """
    Plot the evolution of theory size over time.
    
    Args:
        history: List of theories representing evolution over time
        save_path: Optional path to save the figure
    """
    sizes = [len(theory) for theory in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, marker='o', linestyle='-', linewidth=2, markersize=4)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Theory Size (Number of Clauses)', fontsize=12)
    plt.title('Theory Evolution Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def display_rules(theory: Theory, max_rules: Optional[int] = None) -> None:
    """
    Display learned rules in a readable format.
    
    Args:
        theory: The theory to display
        max_rules: Maximum number of rules to display (None for all)
    """
    clauses = theory.get_clauses()
    
    if max_rules is not None:
        clauses = clauses[:max_rules]
    
    print(f"\n{'='*60}")
    print(f"Learned Theory ({len(clauses)} clauses)")
    print(f"{'='*60}\n")
    
    if len(clauses) == 0:
        print("  (Empty theory)")
        return
    
    for i, clause in enumerate(clauses, 1):
        print(f"Rule {i}:")
        print(f"  {clause}")
        print()


def plot_rule_coverage(
    theory: Theory,
    facts: List,
    oracle,
    save_path: Optional[str] = None
) -> Dict[Clause, int]:
    """
    Plot which rules cover which facts.
    
    Args:
        theory: The learned theory
        facts: List of observed facts
        oracle: The oracle for checking
        save_path: Optional path to save the figure
    
    Returns:
        Dictionary mapping clauses to number of facts they cover
    """
    from ..core.resolution import ResolutionEngine
    
    resolution_engine = ResolutionEngine()
    coverage = {}
    
    for clause in theory.get_clauses():
        count = 0
        for fact in facts:
            # Check if this clause can help prove the fact
            # (simplified - in practice, need to check if clause is used in proof)
            if resolution_engine.can_prove(theory, fact, oracle.query):
                count += 1
        coverage[clause] = count
    
    # Create bar plot
    clauses_str = [str(clause)[:50] + "..." if len(str(clause)) > 50 else str(clause)
                   for clause in coverage.keys()]
    counts = list(coverage.values())
    
    plt.figure(figsize=(12, max(6, len(clauses_str) * 0.5)))
    plt.barh(range(len(clauses_str)), counts)
    plt.yticks(range(len(clauses_str)), clauses_str, fontsize=8)
    plt.xlabel('Number of Facts Covered', fontsize=12)
    plt.title('Rule Coverage', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return coverage


def plot_rule_accuracy(
    theory: Theory,
    test_facts: List,
    oracle,
    save_path: Optional[str] = None
) -> Dict[str, float]:
    """
    Plot accuracy metrics for the learned theory.
    
    Args:
        theory: The learned theory
        test_facts: List of test facts
        oracle: The oracle for checking
        save_path: Optional path to save the figure
    
    Returns:
        Dictionary with accuracy metrics
    """
    from ..core.resolution import ResolutionEngine
    
    resolution_engine = ResolutionEngine()
    
    correct = 0
    total = 0
    
    for fact in test_facts:
        total += 1
        can_prove = resolution_engine.can_prove(theory, fact, oracle.query)
        oracle_says = oracle.query(fact)
        
        if can_prove == oracle_says:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    metrics = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }
    
    # Create bar plot
    plt.figure(figsize=(8, 6))
    plt.bar(['Accuracy'], [accuracy], color='steelblue', alpha=0.7)
    plt.ylim(0, 1.0)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Theory Accuracy: {accuracy:.2%}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add text annotation
    plt.text(0, accuracy + 0.05, f'{correct}/{total} correct', 
             ha='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return metrics


def compare_theories(
    theories: List[Theory],
    labels: List[str],
    save_path: Optional[str] = None
) -> None:
    """
    Compare multiple theories side by side.
    
    Args:
        theories: List of theories to compare
        labels: Labels for each theory
        save_path: Optional path to save the figure
    """
    sizes = [len(theory) for theory in theories]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, sizes, color='steelblue', alpha=0.7)
    plt.ylabel('Theory Size (Number of Clauses)', fontsize=12)
    plt.title('Theory Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

