"""Visualization utilities for rule evolution, display, and proof trees."""

import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Any
from ..core.theory import Theory
from ..core.clauses import Clause
from ..core.atoms import Atom
from ..core.resolution import ResolutionEngine, ProofNode


# ============================================================================
# Proof Tree Visualization
# ============================================================================

class ProofTreeVisualizer:
    """Visualizer for resolution proof trees."""
    
    def __init__(self, resolution_engine: Optional[ResolutionEngine] = None):
        """Initialize with an optional resolution engine."""
        self.resolution_engine = resolution_engine or ResolutionEngine(max_depth=10)
    
    def visualize_proof(
        self,
        theory: Theory,
        goal: Atom,
        oracle=None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10)
    ) -> Optional[ProofNode]:
        """
        Visualize the proof tree for a goal.
        
        Args:
            theory: The theory to prove from
            goal: The goal to prove
            oracle: Optional oracle for checking ground atoms
            save_path: Optional path to save the figure
            figsize: Figure size (width, height)
        
        Returns:
            The proof tree root node if successful, None otherwise
        """
        oracle_func = oracle.query if hasattr(oracle, 'query') else oracle
        proof = self.resolution_engine.prove(theory, goal, oracle=oracle_func)
        
        if proof is None:
            print(f"Could not prove: {goal}")
            return None
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.axis('off')
        
        # Calculate tree layout
        positions = self._calculate_positions(proof)
        
        # Draw the tree
        self._draw_tree(ax, proof, positions)
        
        plt.title(f"Proof Tree for: {goal}", fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        plt.show()
        
        return proof
    
    def _calculate_positions(self, root: ProofNode) -> Dict[int, Tuple[float, float]]:
        """Calculate positions for each node in the tree."""
        positions = {}
        
        # Calculate tree dimensions
        max_depth = self._get_max_depth(root)
        width_at_depth = self._get_width_at_depths(root)
        
        # Assign positions using a recursive approach
        self._assign_positions(root, positions, 0, 0, 1, max_depth)
        
        return positions
    
    def _get_max_depth(self, node: ProofNode) -> int:
        """Get the maximum depth of the tree."""
        if not node.children:
            return node.depth
        return max(self._get_max_depth(child) for child in node.children)
    
    def _get_width_at_depths(self, node: ProofNode) -> Dict[int, int]:
        """Get the number of nodes at each depth level."""
        widths = {}
        self._count_at_depth(node, widths)
        return widths
    
    def _count_at_depth(self, node: ProofNode, widths: Dict[int, int]) -> None:
        """Count nodes at each depth."""
        depth = node.depth
        widths[depth] = widths.get(depth, 0) + 1
        for child in node.children:
            self._count_at_depth(child, widths)
    
    def _assign_positions(
        self,
        node: ProofNode,
        positions: Dict[int, Tuple[float, float]],
        node_id: int,
        x_min: float,
        x_max: float,
        max_depth: int
    ) -> int:
        """Assign positions to nodes."""
        # Y position based on depth
        y = 0.9 - (node.depth / max(max_depth, 1)) * 1.6
        
        # X position centered in the allocated space
        x = (x_min + x_max) / 2
        
        positions[id(node)] = (x, y)
        
        # Assign positions to children
        if node.children:
            child_width = (x_max - x_min) / len(node.children)
            for i, child in enumerate(node.children):
                child_x_min = x_min + i * child_width
                child_x_max = x_min + (i + 1) * child_width
                self._assign_positions(child, positions, node_id + 1, 
                                       child_x_min, child_x_max, max_depth)
        
        return node_id
    
    def _draw_tree(self, ax, node: ProofNode, positions: Dict[int, Tuple[float, float]]) -> None:
        """Draw the proof tree on the axes."""
        node_pos = positions[id(node)]
        
        # Draw edges to children first (so they're behind nodes)
        for child in node.children:
            child_pos = positions[id(child)]
            ax.plot([node_pos[0], child_pos[0]], [node_pos[1], child_pos[1]], 
                   'k-', linewidth=1.5, zorder=1)
        
        # Draw this node
        self._draw_node(ax, node, node_pos)
        
        # Recursively draw children
        for child in node.children:
            self._draw_tree(ax, child, positions)
    
    def _draw_node(self, ax, node: ProofNode, pos: Tuple[float, float]) -> None:
        """Draw a single node."""
        x, y = pos
        
        # Determine node color based on type
        if node.clause is not None and node.clause.is_fact():
            color = '#90EE90'  # Light green for facts
            edge_color = '#228B22'
        elif node.clause is not None:
            color = '#87CEEB'  # Sky blue for rules
            edge_color = '#4169E1'
        else:
            color = '#FFE4B5'  # Moccasin for goals
            edge_color = '#DAA520'
        
        # Create node text
        goal_str = str(node.goal.apply_substitution(node.substitution))
        if len(goal_str) > 30:
            goal_str = goal_str[:27] + "..."
        
        # Draw node box
        bbox = dict(boxstyle='round,pad=0.3', facecolor=color, 
                   edgecolor=edge_color, linewidth=2)
        ax.text(x, y, goal_str, ha='center', va='center', fontsize=9,
               bbox=bbox, zorder=2)
        
        # Add clause label if applicable
        if node.clause is not None:
            clause_str = str(node.clause)
            if len(clause_str) > 40:
                clause_str = clause_str[:37] + "..."
            ax.text(x, y - 0.08, f"via: {clause_str}", ha='center', va='top',
                   fontsize=7, color='gray', style='italic')
    
    def print_proof_tree(self, proof: ProofNode, indent: int = 0) -> str:
        """
        Generate a text representation of a proof tree.
        
        Args:
            proof: The root of the proof tree
            indent: Current indentation level
        
        Returns:
            String representation of the proof tree
        """
        lines = []
        prefix = "  " * indent
        
        # Format the goal
        goal_str = str(proof.goal.apply_substitution(proof.substitution))
        
        if proof.clause is not None:
            clause_type = "FACT" if proof.clause.is_fact() else "RULE"
            lines.append(f"{prefix}├── {goal_str}")
            lines.append(f"{prefix}│   └── [{clause_type}] {proof.clause}")
        else:
            lines.append(f"{prefix}├── {goal_str} (goal)")
        
        for i, child in enumerate(proof.children):
            is_last = (i == len(proof.children) - 1)
            child_lines = self.print_proof_tree(child, indent + 1)
            lines.append(child_lines)
        
        return "\n".join(lines)


def visualize_proof_tree(
    theory: Theory,
    goal: Atom,
    oracle=None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> Optional[ProofNode]:
    """
    Convenience function to visualize a proof tree.
    
    Args:
        theory: The theory to prove from
        goal: The goal to prove
        oracle: Optional oracle for checking ground atoms
        save_path: Optional path to save the figure
        figsize: Figure size
    
    Returns:
        The proof tree if successful
    """
    visualizer = ProofTreeVisualizer()
    return visualizer.visualize_proof(theory, goal, oracle, save_path, figsize)


def print_proof_tree(theory: Theory, goal: Atom, oracle=None) -> None:
    """
    Print a text representation of a proof tree.
    
    Args:
        theory: The theory to prove from
        goal: The goal to prove
        oracle: Optional oracle for checking ground atoms
    """
    engine = ResolutionEngine(max_depth=10)
    oracle_func = oracle.query if hasattr(oracle, 'query') else oracle
    proof = engine.prove(theory, goal, oracle=oracle_func)
    
    if proof is None:
        print(f"Could not prove: {goal}")
        return
    
    print(f"\nProof Tree for: {goal}")
    print("=" * 60)
    
    visualizer = ProofTreeVisualizer(engine)
    print(visualizer.print_proof_tree(proof))
    print("=" * 60)


def visualize_all_proofs(
    theory: Theory,
    facts: List[Atom],
    oracle=None,
    max_proofs: int = 5,
    save_dir: Optional[str] = None
) -> List[ProofNode]:
    """
    Visualize proofs for multiple facts.
    
    Args:
        theory: The theory to prove from
        facts: List of facts to prove
        oracle: Optional oracle
        max_proofs: Maximum number of proofs to visualize
        save_dir: Optional directory to save figures
    
    Returns:
        List of successful proof trees
    """
    proofs = []
    visualizer = ProofTreeVisualizer()
    
    for i, fact in enumerate(facts[:max_proofs]):
        save_path = f"{save_dir}/proof_{i}.png" if save_dir else None
        proof = visualizer.visualize_proof(theory, fact, oracle, save_path)
        if proof is not None:
            proofs.append(proof)
    
    return proofs


# ============================================================================
# Theory Evolution Visualization
# ============================================================================

def plot_theory_evolution(history: List[Theory], save_path: Optional[str] = None) -> None:
    """
    Plot the evolution of theory size over time.
    
    Args:
        history: List of theories representing evolution over time
        save_path: Optional path to save the figure
    """
    sizes = [len(theory) for theory in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, marker='o', linestyle='-', linewidth=2, markersize=4, color='#2E86AB')
    plt.fill_between(range(len(sizes)), sizes, alpha=0.3, color='#2E86AB')
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


