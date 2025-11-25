"""Main incremental Algorithm 2 from Shapiro (1981)."""

from typing import List, Iterator, Optional, Callable
from ..core.atoms import Atom, Variable, Constant
from ..core.clauses import Clause
from ..core.theory import Theory
from ..core.resolution import ResolutionEngine
from .oracle import Oracle
from .backtracing import ContradictionBacktracer
from .refinement import RefinementOperator, CompositeRefinementOperator, AddConditionRefinement, SpecializePredicateRefinement


class ModelInference:
    """
    Implements Shapiro's incremental Algorithm 2 for model inference.
    
    Algorithm:
    1. Start with empty theory T
    2. For each new fact:
       a. While T is too strong (proves false facts):
          - Find contradiction via backtracing
          - Remove false hypothesis
       b. While T is too weak (cannot prove true fact):
          - Generate refinements
          - Add plausible hypotheses
    3. Output T
    """
    
    def __init__(
        self,
        oracle: Oracle,
        refinement_operator: Optional[RefinementOperator] = None,
        max_iterations: int = 100,
        max_theory_size: int = 50
    ):
        """
        Initialize the model inference system.
        
        Args:
            oracle: The oracle for querying the model
            refinement_operator: Optional refinement operator (default: composite)
            max_iterations: Maximum iterations per fact
            max_theory_size: Maximum number of clauses in theory
        """
        self.oracle = oracle
        self.resolution_engine = ResolutionEngine(max_depth=10)
        self.backtracer = ContradictionBacktracer(self.resolution_engine)
        self.max_iterations = max_iterations
        self.max_theory_size = max_theory_size
        
        # Use default refinement operator if none provided
        if refinement_operator is None:
            # Will be set up when we know feature names
            self.refinement_operator = None
        else:
            self.refinement_operator = refinement_operator
        
        # History for tracking
        self.history: List[Theory] = []
        self.observed_facts: List[Atom] = []
    
    def infer_theory(self, fact_stream: Iterator[Atom]) -> Theory:
        """
        Infer a theory from a stream of facts.
        
        Args:
            fact_stream: Iterator of observed facts (ground atoms)
        
        Returns:
            The inferred theory
        """
        theory = Theory()
        self.history = [theory.copy()]
        self.observed_facts = []
        
        for fact in fact_stream:
            if not fact.is_ground():
                continue  # Skip non-ground facts
            
            self.observed_facts.append(fact)
            
            # Set up refinement operator if needed
            if self.refinement_operator is None:
                self._setup_default_refinement_operator()
            
            # Process this fact
            theory = self._process_fact(theory, fact)
            self.history.append(theory.copy())
        
        return theory
    
    def _process_fact(self, theory: Theory, fact: Atom) -> Theory:
        """Process a single fact through the algorithm."""
        iterations = 0
        
        while iterations < self.max_iterations:
            iterations += 1
            
            # Step 1: Remove contradictions (theory too strong)
            while self._is_too_strong(theory, [fact]):
                false_clause = self.backtracer.find_contradiction(theory, fact, self.oracle)
                if false_clause is not None:
                    theory.remove_clause(false_clause)
                    self.history.append(theory.copy())
                else:
                    break  # No more contradictions found
            
            # Step 2: Add refinements (theory too weak)
            if self._is_too_weak(theory, fact):
                refinements = self._generate_refinements(theory, fact)
                
                # Add refinements that are consistent
                for refinement in refinements:
                    if len(theory) >= self.max_theory_size:
                        break
                    
                    # Check if refinement is consistent (doesn't immediately contradict)
                    if self._is_plausible(refinement, fact):
                        theory.add_clause(refinement)
                        self.history.append(theory.copy())
                else:
                    continue
                break
            else:
                # Theory is neither too strong nor too weak - done with this fact
                break
        
        return theory
    
    def _is_too_strong(self, theory: Theory, facts: List[Atom]) -> bool:
        """Check if theory is too strong (proves false facts)."""
        return self.backtracer.is_too_strong(theory, facts, self.oracle)
    
    def _is_too_weak(self, theory: Theory, fact: Atom) -> bool:
        """Check if theory is too weak (cannot prove true fact)."""
        # If oracle says fact is false, theory shouldn't prove it (not too weak)
        if not self.oracle.query(fact):
            return False
        
        # Check if theory can prove the fact
        can_prove = self.resolution_engine.can_prove(theory, fact, oracle=self.oracle.query)
        return not can_prove
    
    def _generate_refinements(self, theory: Theory, fact: Atom) -> List[Clause]:
        """Generate refinements to make theory less weak."""
        if self.refinement_operator is None:
            return []
        
        refinements = []
        
        # Generate initial hypothesis if theory is empty
        if len(theory) == 0:
            initial_hypothesis = self._generate_initial_hypothesis(fact)
            if initial_hypothesis is not None:
                refinements.append(initial_hypothesis)
        else:
            # Refine existing clauses
            for clause in theory.get_clauses():
                clause_refinements = self.refinement_operator.refine(
                    clause, self.oracle, self.observed_facts
                )
                refinements.extend(clause_refinements)
        
        return refinements
    
    def _generate_initial_hypothesis(self, fact: Atom) -> Optional[Clause]:
        """Generate an initial hypothesis for a fact."""
        # For predict(instance_id, label), generate a simple rule
        if fact.predicate == "predict" and len(fact.arguments) == 2:
            instance_id_arg, label_arg = fact.arguments
            
            if isinstance(instance_id_arg, Constant) and isinstance(label_arg, Constant):
                # Create rule: predict(X, label) :- feature(X, F, V)
                X = Variable("X")
                head = Atom("predict", [X, label_arg])
                
                # Try to find a feature that might explain this
                instance_id = instance_id_arg.value
                # This is a simple initial hypothesis - will be refined
                return Clause(head, [])
        
        return None
    
    def _is_plausible(self, clause: Clause, fact: Atom) -> bool:
        """Check if a clause is plausible (doesn't immediately contradict)."""
        # Simple check: if clause head unifies with fact, it's plausible
        from ..core.atoms import unify_atoms
        unifier = unify_atoms(clause.head, fact)
        return unifier is not None
    
    def _setup_default_refinement_operator(self) -> None:
        """Set up default refinement operator based on observed facts."""
        # Extract feature names from observed facts
        feature_names = set()
        feature_values = {}
        
        for fact in self.observed_facts:
            if fact.predicate == "feature" and len(fact.arguments) >= 2:
                if isinstance(fact.arguments[1], Constant):
                    feature_name = fact.arguments[1].value
                    feature_names.add(feature_name)
                    
                    if len(fact.arguments) >= 3 and isinstance(fact.arguments[2], Constant):
                        value = fact.arguments[2].value
                        if feature_name not in feature_values:
                            feature_values[feature_name] = set()
                        feature_values[feature_name].add(value)
        
        # Convert sets to lists
        feature_names_list = list(feature_names)
        feature_values_dict = {
            name: list(values) for name, values in feature_values.items()
        }
        
        # Create composite refinement operator
        add_condition = AddConditionRefinement(feature_names_list, feature_values_dict)
        specialize = SpecializePredicateRefinement()
        
        self.refinement_operator = CompositeRefinementOperator([add_condition, specialize])
    
    def get_history(self) -> List[Theory]:
        """Get the history of theory evolution."""
        return self.history.copy()

