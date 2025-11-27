"""SLD resolution engine with proof tree construction."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Callable
from .atoms import Atom, Variable, Term, unify_atoms, compose_substitutions
from .clauses import Clause
from .theory import Theory


@dataclass
class ProofNode:
    """A node in a resolution proof tree."""
    goal: Atom
    substitution: Dict[Variable, Term]
    clause: Optional[Clause] = None  # The clause used to resolve this goal
    parent: Optional['ProofNode'] = None
    children: List['ProofNode'] = None
    depth: int = 0
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.children) == 0
    
    def is_root(self) -> bool:
        """Check if this is the root node (no parent)."""
        return self.parent is None
    
    def get_path_to_root(self) -> List['ProofNode']:
        """Get the path from this node to the root."""
        path = [self]
        current = self.parent
        while current is not None:
            path.append(current)
            current = current.parent
        return path
    
    def get_all_leaf_clauses(self) -> List[Clause]:
        """Get all clauses used at leaf nodes in the subtree rooted at this node."""
        clauses = []
        if self.is_leaf() and self.clause is not None:
            clauses.append(self.clause)
        for child in self.children:
            clauses.extend(child.get_all_leaf_clauses())
        return clauses


class ResolutionEngine:
    """SLD resolution engine for proving goals from a theory."""
    
    def __init__(self, max_depth: int = 10):
        """Initialize the resolution engine with a maximum proof depth."""
        self.max_depth = max_depth
    
    def prove(self, theory: Theory, goal: Atom, oracle: Optional[Callable[[Atom], bool]] = None) -> Optional[ProofNode]:
        """
        Attempt to prove a goal from the theory using SLD resolution.
        
        Args:
            theory: The theory to prove from
            goal: The goal atom to prove
            oracle: Optional oracle function to check ground atoms (for model inference)
        
        Returns:
            A ProofNode representing the proof tree if successful, None otherwise
        """
        root = ProofNode(goal, {}, depth=0)
        success = self._prove_recursive(theory, root, oracle)
        return root if success else None
    
    def _prove_recursive(
        self,
        theory: Theory,
        node: ProofNode,
        oracle: Optional[Callable[[Atom], bool]] = None
    ) -> bool:
        """
        Recursively attempt to prove a goal.
        
        Returns True if the goal is provable, False otherwise.
        """
        if node.depth > self.max_depth:
            return False
        
        goal = node.goal
        substitution = node.substitution
        
        # Apply current substitution to goal
        goal_substituted = goal.apply_substitution(substitution)
        
        # If goal is ground and we have an oracle, check it directly
        if oracle is not None and goal_substituted.is_ground():
            if oracle(goal_substituted):
                # Success - this is a leaf node
                return True
            # If oracle says false, we can't prove it (but continue trying rules)
        
        # Try to resolve with clauses in the theory
        matching_clauses = theory.find_clauses_with_head(goal_substituted)
        
        for clause in matching_clauses:
            # Standardize apart to avoid variable conflicts
            clause_standardized = clause.standardize_apart()
            
            # Try to unify goal with clause head
            unifier = unify_atoms(goal_substituted, clause_standardized.head)
            if unifier is None:
                continue
            
            # Compose substitutions
            new_substitution = compose_substitutions(substitution, unifier)
            
            if clause_standardized.is_fact():
                # Fact clause - we've proven this goal
                child = ProofNode(
                    goal=goal_substituted,
                    substitution=new_substitution,
                    clause=clause_standardized,
                    parent=node,
                    depth=node.depth + 1
                )
                node.children.append(child)
                return True
            else:
                # Rule clause - need to prove body
                # Create child nodes for each body atom
                child_node = ProofNode(
                    goal=goal_substituted,
                    substitution=new_substitution,
                    clause=clause_standardized,
                    parent=node,
                    depth=node.depth + 1
                )
                node.children.append(child_node)
                
                # Try to prove each body atom
                all_proven = True
                for body_atom in clause_standardized.body:
                    body_substituted = body_atom.apply_substitution(new_substitution)
                    body_node = ProofNode(
                        goal=body_substituted,
                        substitution=new_substitution,
                        parent=child_node,
                        depth=node.depth + 2
                    )
                    child_node.children.append(body_node)
                    
                    if not self._prove_recursive(theory, body_node, oracle):
                        all_proven = False
                        break
                
                if all_proven:
                    return True
        
        # If we have an oracle and goal is ground, and no rules matched, return False
        if oracle is not None and goal_substituted.is_ground():
            return False
        
        # Could not prove
        return False
    
    def find_contradiction(
        self,
        theory: Theory,
        fact: Atom,
        oracle: Callable[[Atom], bool]
    ) -> Optional[ProofNode]:
        """
        Find a contradiction: theory proves fact, but oracle says fact is false.
        
        Returns the proof node if contradiction found, None otherwise.
        """
        # Check if oracle says fact is false
        if oracle(fact):
            return None  # No contradiction - oracle agrees
        
        # Try to prove fact from theory
        proof = self.prove(theory, fact, oracle=None)  # Don't use oracle in proof
        
        if proof is not None:
            # Theory proves it, but oracle says false - contradiction!
            return proof
        
        return None
    
    def can_prove(self, theory: Theory, goal: Atom, oracle: Optional[Callable[[Atom], bool]] = None) -> bool:
        """Check if a goal can be proven from the theory."""
        proof = self.prove(theory, goal, oracle)
        return proof is not None and self._has_successful_proof(proof)
    
    def _has_successful_proof(self, node: ProofNode) -> bool:
        """Check if a proof node represents a successful proof."""
        # A successful proof has at least one path to a leaf that represents success
        if node.is_leaf():
            # Leaf node - check if it's a fact or oracle-confirmed
            return node.clause is not None and (node.clause.is_fact() or len(node.children) == 0)
        
        # Check if any child has a successful proof
        for child in node.children:
            if self._has_successful_proof(child):
                return True
        
        return False


