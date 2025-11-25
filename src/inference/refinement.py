"""Refinement operators for generating new hypotheses."""

from abc import ABC, abstractmethod
from typing import List, Set, Optional, Dict
from ..core.atoms import Atom, Variable, Constant
from ..core.clauses import Clause
from .oracle import Oracle


class RefinementOperator(ABC):
    """Abstract base class for refinement operators."""
    
    @abstractmethod
    def refine(self, clause: Clause, oracle: Oracle, observed_facts: List[Atom]) -> List[Clause]:
        """
        Generate refinements of a clause.
        
        Args:
            clause: The clause to refine
            oracle: The oracle for querying
            observed_facts: List of observed facts to guide refinement
        
        Returns:
            List of refined clauses
        """
        pass


class AddConditionRefinement(RefinementOperator):
    """
    Refinement operator that adds conditions to clause bodies.
    
    For a clause predict(X, L) :- body, generates:
    - predict(X, L) :- body, feature(X, F, V)
    - predict(X, L) :- body, has_feature(X, F)
    """
    
    def __init__(self, feature_names: List[str], feature_values: Optional[Dict[str, List[str]]] = None):
        """
        Initialize the refinement operator.
        
        Args:
            feature_names: List of feature names to use in refinements
            feature_values: Optional mapping of feature_name -> list of possible values
        """
        self.feature_names = feature_names
        self.feature_values = feature_values or {}
    
    def refine(self, clause: Clause, oracle: Oracle, observed_facts: List[Atom]) -> List[Clause]:
        """Generate refinements by adding conditions."""
        refinements = []
        
        # Get the variable used for the instance (usually first argument of head)
        if not clause.head.arguments:
            return refinements
        
        instance_var = clause.head.arguments[0]
        if not isinstance(instance_var, Variable):
            return refinements
        
        # Get existing body atoms to avoid duplicates
        existing_predicates = set()
        for atom in clause.body:
            existing_predicates.add((atom.predicate, len(atom.arguments)))
        
        # Add feature conditions
        for feature_name in self.feature_names:
            # Add feature(instance_var, feature_name, value) for each possible value
            if feature_name in self.feature_values:
                for value in self.feature_values[feature_name]:
                    new_atom = Atom(
                        "feature",
                        [instance_var, Constant(feature_name), Constant(value)]
                    )
                    if (new_atom.predicate, len(new_atom.arguments)) not in existing_predicates:
                        new_body = clause.body + [new_atom]
                        new_clause = Clause(clause.head, new_body)
                        refinements.append(new_clause)
            
            # Add has_feature(instance_var, feature_name)
            new_atom = Atom(
                "has_feature",
                [instance_var, Constant(feature_name)]
            )
            if (new_atom.predicate, len(new_atom.arguments)) not in existing_predicates:
                new_body = clause.body + [new_atom]
                new_clause = Clause(clause.head, new_body)
                refinements.append(new_clause)
        
        return refinements


class SpecializePredicateRefinement(RefinementOperator):
    """
    Refinement operator that specializes predicate arguments.
    
    For feature(X, F, V), generates feature(X, F, specific_value) for each observed value.
    """
    
    def refine(self, clause: Clause, oracle: Oracle, observed_facts: List[Atom]) -> List[Clause]:
        """Generate refinements by specializing predicates."""
        refinements = []
        
        # Find feature atoms in the body with variables
        for i, atom in enumerate(clause.body):
            if atom.predicate == "feature" and len(atom.arguments) == 3:
                feature_name_arg = atom.arguments[1]
                value_arg = atom.arguments[2]
                
                # If value is a variable, specialize it
                if isinstance(value_arg, Variable):
                    # Find all observed values for this feature
                    observed_values = self._get_observed_values(
                        feature_name_arg, observed_facts, oracle
                    )
                    
                    for value in observed_values:
                        new_body = clause.body.copy()
                        new_atom = Atom(
                            atom.predicate,
                            [
                                atom.arguments[0],
                                atom.arguments[1],
                                Constant(value)
                            ]
                        )
                        new_body[i] = new_atom
                        new_clause = Clause(clause.head, new_body)
                        refinements.append(new_clause)
        
        return refinements
    
    def _get_observed_values(
        self,
        feature_name_arg: Atom,
        observed_facts: List[Atom],
        oracle: Oracle
    ) -> Set[str]:
        """Get all observed values for a feature from observed facts."""
        values = set()
        
        # Extract feature name if it's a constant
        if isinstance(feature_name_arg, Constant):
            feature_name = feature_name_arg.value
        else:
            return values
        
        # Look through observed facts
        for fact in observed_facts:
            if fact.predicate == "feature" and len(fact.arguments) == 3:
                if (isinstance(fact.arguments[1], Constant) and
                    fact.arguments[1].value == feature_name and
                    isinstance(fact.arguments[2], Constant)):
                    values.add(fact.arguments[2].value)
        
        return values


class ConjunctionRefinement(RefinementOperator):
    """
    Refinement operator that adds conjunctions of existing conditions.
    
    Generates clauses with multiple feature conditions.
    """
    
    def __init__(self, max_conjunctions: int = 2):
        """Initialize with maximum number of conjunctions."""
        self.max_conjunctions = max_conjunctions
    
    def refine(self, clause: Clause, oracle: Oracle, observed_facts: List[Atom]) -> List[Clause]:
        """Generate refinements by adding conjunctions."""
        refinements = []
        
        # Get existing feature atoms
        feature_atoms = [
            atom for atom in clause.body
            if atom.predicate in ("feature", "has_feature")
        ]
        
        if len(feature_atoms) >= self.max_conjunctions:
            return refinements  # Already has enough conditions
        
        # Add combinations of feature atoms
        # This is a simplified version - in practice, you'd want smarter selection
        for fact in observed_facts[:10]:  # Limit to avoid explosion
            if fact.predicate in ("feature", "has_feature"):
                if fact not in clause.body:
                    new_body = clause.body + [fact]
                    new_clause = Clause(clause.head, new_body)
                    refinements.append(new_clause)
        
        return refinements


class CompositeRefinementOperator(RefinementOperator):
    """Composite refinement operator that combines multiple operators."""
    
    def __init__(self, operators: List[RefinementOperator]):
        """Initialize with a list of refinement operators."""
        self.operators = operators
    
    def refine(self, clause: Clause, oracle: Oracle, observed_facts: List[Atom]) -> List[Clause]:
        """Apply all refinement operators and combine results."""
        all_refinements = []
        seen = set()
        
        for operator in self.operators:
            refinements = operator.refine(clause, oracle, observed_facts)
            for ref in refinements:
                if ref not in seen:
                    all_refinements.append(ref)
                    seen.add(ref)
        
        return all_refinements

