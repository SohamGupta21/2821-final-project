"""Contradiction backtracing algorithm."""

from typing import Optional, List
from ..core.atoms import Atom
from ..core.clauses import Clause
from ..core.theory import Theory
from ..core.resolution import ResolutionEngine, ProofNode
from .oracle import Oracle


class ContradictionBacktracer:
    """
    Implements contradiction backtracing to find false hypotheses.
    
    Based on Shapiro (1981), Section 2: Contradiction Backtracing.
    """
    
    def __init__(self, resolution_engine: Optional[ResolutionEngine] = None):
        """Initialize the backtracer with an optional resolution engine."""
        self.resolution_engine = resolution_engine or ResolutionEngine(max_depth=10)
    
    def find_contradiction(
        self,
        theory: Theory,
        fact: Atom,
        oracle: Oracle
    ) -> Optional[Clause]:
        """
        Find a contradiction: theory proves fact, but oracle says fact is false.
        
        Returns the clause that should be removed, or None if no contradiction found.
        
        Args:
            theory: The current theory
            fact: A fact that the theory proves but oracle says is false
            oracle: The oracle to query
        
        Returns:
            The clause causing the contradiction, or None
        """
        # Check if oracle says fact is false
        if oracle.query(fact):
            return None  # No contradiction - oracle agrees
        
        # Try to prove fact from theory
        proof = self.resolution_engine.prove(theory, fact, oracle=None)
        
        if proof is None:
            return None  # Theory doesn't prove it, so no contradiction
        
        # Theory proves it, but oracle says false - contradiction!
        # Backtrace to find the false hypothesis
        return self._backtrace_contradiction(proof, oracle)
    
    def _backtrace_contradiction(self, proof: ProofNode, oracle: Oracle) -> Optional[Clause]:
        """
        Backtrace through the proof tree to find the false hypothesis.
        
        Algorithm: Walk down the proof tree, at each node checking if the
        goal is true according to the oracle. When we find a node where
        the goal is false but was proven by a clause, that clause is the
        false hypothesis.
        """
        # Start from root and work down
        return self._backtrace_recursive(proof, oracle)
    
    def _backtrace_recursive(self, node: ProofNode, oracle: Oracle) -> Optional[Clause]:
        """
        Recursively backtrace through the proof tree.
        
        Returns the false clause if found, None otherwise.
        """
        # Check if this goal is ground and false according to oracle
        goal = node.goal
        goal_ground = goal.apply_substitution(node.substitution)
        
        if goal_ground.is_ground():
            if not oracle.query(goal_ground):
                # This goal is false - the clause that proved it must be false
                if node.clause is not None:
                    return node.clause
        
        # Recursively check children
        for child in node.children:
            result = self._backtrace_recursive(child, oracle)
            if result is not None:
                return result
        
        # If this node has a clause and we've checked all children,
        # the clause might be the problem
        if node.clause is not None and node.is_leaf():
            # This is a leaf clause - check if it's false
            # A leaf clause should be a fact that's directly queried
            if goal_ground.is_ground() and not oracle.query(goal_ground):
                return node.clause
        
        return None
    
    def find_all_contradictions(
        self,
        theory: Theory,
        facts: List[Atom],
        oracle: Oracle
    ) -> List[Clause]:
        """
        Find all clauses causing contradictions with the given facts.
        
        Args:
            theory: The current theory
            facts: List of facts to check
            oracle: The oracle to query
        
        Returns:
            List of clauses that cause contradictions
        """
        false_clauses = []
        seen_clauses = set()
        
        for fact in facts:
            false_clause = self.find_contradiction(theory, fact, oracle)
            if false_clause is not None and false_clause not in seen_clauses:
                false_clauses.append(false_clause)
                seen_clauses.add(false_clause)
        
        return false_clauses
    
    def is_too_strong(
        self,
        theory: Theory,
        facts: List[Atom],
        oracle: Oracle
    ) -> bool:
        """
        Check if the theory is too strong (proves facts that are false).
        
        Args:
            theory: The current theory
            facts: List of observed facts
            oracle: The oracle to query
        
        Returns:
            True if theory is too strong, False otherwise
        """
        for fact in facts:
            # Check if theory proves this fact
            can_prove = self.resolution_engine.can_prove(theory, fact, oracle=None)
            if can_prove:
                # Check if oracle says it's false
                if not oracle.query(fact):
                    return True  # Theory proves false fact - too strong!
        return False

