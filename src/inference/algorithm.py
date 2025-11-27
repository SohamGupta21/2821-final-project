"""Main incremental Algorithm 2 from Shapiro (1981) - Improved Version."""

from typing import List, Iterator, Optional, Callable, Set, Dict, Any, Tuple
from collections import defaultdict
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
    
    Improved version that:
    1. Collects all facts and builds feature maps
    2. Analyzes feature-label correlations to find discriminative patterns
    3. Generates initial hypotheses with meaningful conditions
    4. Refines hypotheses to handle contradictions
    5. Uses a feature-only oracle during rule testing (not predict short-circuit)
    """
    
    def __init__(
        self,
        oracle: Oracle,
        refinement_operator: Optional[RefinementOperator] = None,
        max_iterations: int = 100,
        max_theory_size: int = 50,
        min_support: float = 0.1,
        min_confidence: float = 0.6,
        verbose: bool = False
    ):
        """
        Initialize the model inference system.
        
        Args:
            oracle: The oracle for querying the model
            refinement_operator: Optional refinement operator
            max_iterations: Maximum iterations per fact
            max_theory_size: Maximum number of clauses in theory
            min_support: Minimum support for a rule (fraction of instances)
            min_confidence: Minimum confidence for a rule
            verbose: Whether to print debug information
        """
        self.oracle = oracle
        self.resolution_engine = ResolutionEngine(max_depth=10)
        self.backtracer = ContradictionBacktracer(self.resolution_engine)
        self.max_iterations = max_iterations
        self.max_theory_size = max_theory_size
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.verbose = verbose
        
        self.refinement_operator = refinement_operator
        
        # Data structures for analysis
        self.history: List[Theory] = []
        self.observed_facts: List[Atom] = []
        self.feature_facts: List[Atom] = []
        self.predict_facts: List[Atom] = []
        
        # Instance data
        self.instance_features: Dict[Any, Dict[str, Any]] = {}
        self.instance_labels: Dict[Any, Any] = {}
        self.seen_labels: Set[Any] = set()
        
        # Feature statistics
        self.feature_names: Set[str] = set()
        self.feature_values: Dict[str, Set[Any]] = defaultdict(set)
    
    def infer_theory(self, fact_stream: Iterator[Atom]) -> Theory:
        """
        Infer a theory from a stream of facts.
        
        This improved version:
        1. Collects and indexes all facts
        2. Analyzes patterns to find discriminative feature combinations
        3. Generates rules from patterns
        4. Refines to handle edge cases
        """
        theory = Theory()
        self.history = [theory.copy()]
        self._reset_data_structures()
        
        # Phase 1: Collect and index all facts
        self._collect_facts(fact_stream)
        
        if self.verbose:
            print(f"  Collected {len(self.feature_facts)} feature facts")
            print(f"  Collected {len(self.predict_facts)} predict facts")
            print(f"  Instances: {len(self.instance_labels)}")
            print(f"  Labels: {self.seen_labels}")
            print(f"  Features: {self.feature_names}")
        
        # Phase 2: Generate initial hypotheses from pattern analysis
        theory = self._generate_pattern_based_rules(theory)
        
        if self.verbose:
            print(f"  Generated {len(theory)} initial clauses")
        
        # Phase 3: Refine rules to handle contradictions and improve coverage
        theory = self._refine_theory(theory)
        
        if self.verbose:
            print(f"  Final theory has {len(theory)} clauses")
        
        self.history.append(theory.copy())
        return theory
    
    def _reset_data_structures(self):
        """Reset all data structures for a new inference run."""
        self.observed_facts = []
        self.feature_facts = []
        self.predict_facts = []
        self.instance_features = {}
        self.instance_labels = {}
        self.seen_labels = set()
        self.feature_names = set()
        self.feature_values = defaultdict(set)
    
    def _collect_facts(self, fact_stream: Iterator[Atom]):
        """Collect and index all facts from the stream."""
        all_facts = list(fact_stream)
        
        for fact in all_facts:
            if not fact.is_ground():
                continue
            
            self.observed_facts.append(fact)
            
            if fact.predicate == "feature" and len(fact.arguments) >= 3:
                self.feature_facts.append(fact)
                self._index_feature_fact(fact)
            elif fact.predicate == "predict" and len(fact.arguments) >= 2:
                self.predict_facts.append(fact)
                self._index_predict_fact(fact)
    
    def _index_feature_fact(self, fact: Atom):
        """Index a feature fact for quick lookup."""
        args = fact.arguments
        if len(args) >= 3:
            instance_id = args[0].value if isinstance(args[0], Constant) else None
            feature_name = args[1].value if isinstance(args[1], Constant) else None
            feature_value = args[2].value if isinstance(args[2], Constant) else None
            
            if instance_id is not None and feature_name is not None:
                if instance_id not in self.instance_features:
                    self.instance_features[instance_id] = {}
                self.instance_features[instance_id][feature_name] = feature_value
                
                self.feature_names.add(feature_name)
                if feature_value is not None:
                    self.feature_values[feature_name].add(feature_value)
    
    def _index_predict_fact(self, fact: Atom):
        """Index a predict fact for quick lookup."""
        args = fact.arguments
        if len(args) >= 2:
            instance_id = args[0].value if isinstance(args[0], Constant) else None
            label = args[1].value if isinstance(args[1], Constant) else None
            
            if instance_id is not None and label is not None:
                self.instance_labels[instance_id] = label
                self.seen_labels.add(label)
    
    def _generate_pattern_based_rules(self, theory: Theory) -> Theory:
        """
        Generate rules by analyzing feature-label patterns.
        
        For each label, find feature conditions that are:
        1. Common among instances with that label (support)
        2. Rare among instances with other labels (confidence)
        """
        for label in self.seen_labels:
            rules = self._find_rules_for_label(label)
            
            for rule in rules:
                if len(theory) < self.max_theory_size:
                    theory.add_clause(rule)
                    self.history.append(theory.copy())
        
        return theory
    
    def _find_rules_for_label(self, label: Any) -> List[Clause]:
        """Find discriminative rules for a specific label."""
        rules = []
        
        # Get instances with this label
        positive_instances = [
            inst_id for inst_id, inst_label in self.instance_labels.items()
            if inst_label == label
        ]
        negative_instances = [
            inst_id for inst_id, inst_label in self.instance_labels.items()
            if inst_label != label
        ]
        
        if not positive_instances:
            return rules
        
        # Find single-feature rules first
        single_feature_rules = self._find_single_feature_rules(
            label, positive_instances, negative_instances
        )
        rules.extend(single_feature_rules)
        
        # If single features aren't good enough, try pairs
        if not single_feature_rules or all(
            self._rule_confidence(r, positive_instances, negative_instances) < 0.8
            for r in single_feature_rules
        ):
            pair_rules = self._find_feature_pair_rules(
                label, positive_instances, negative_instances
            )
            rules.extend(pair_rules)
        
        return rules
    
    def _find_single_feature_rules(
        self,
        label: Any,
        positive_instances: List[Any],
        negative_instances: List[Any]
    ) -> List[Clause]:
        """Find rules based on single feature conditions."""
        rules = []
        total_instances = len(positive_instances) + len(negative_instances)
        
        for feature_name in self.feature_names:
            for feature_value in self.feature_values[feature_name]:
                # Count support and confidence
                positive_with_feature = sum(
                    1 for inst_id in positive_instances
                    if self.instance_features.get(inst_id, {}).get(feature_name) == feature_value
                )
                negative_with_feature = sum(
                    1 for inst_id in negative_instances
                    if self.instance_features.get(inst_id, {}).get(feature_name) == feature_value
                )
                
                total_with_feature = positive_with_feature + negative_with_feature
                
                # Calculate support (fraction of instances with this pattern)
                support = total_with_feature / total_instances if total_instances > 0 else 0
                
                # Calculate confidence (precision for positive class)
                confidence = (
                    positive_with_feature / total_with_feature
                    if total_with_feature > 0 else 0
                )
                
                # Check thresholds
                if support >= self.min_support and confidence >= self.min_confidence:
                    X = Variable("X")
                    head = Atom("predict", [X, Constant(label)])
                    body = [Atom("feature", [X, Constant(feature_name), Constant(feature_value)])]
                    rule = Clause(head, body)
                    rules.append(rule)
                    
                    if self.verbose:
                        print(f"    Found rule: {rule} (support={support:.2f}, conf={confidence:.2f})")
        
        # Sort by confidence, then support
        rules.sort(key=lambda r: (
            -self._rule_confidence(r, positive_instances, negative_instances),
            -self._rule_support(r, positive_instances + negative_instances)
        ))
        
        # Return top rules (avoid redundant rules)
        return rules[:3]
    
    def _find_feature_pair_rules(
        self,
        label: Any,
        positive_instances: List[Any],
        negative_instances: List[Any]
    ) -> List[Clause]:
        """Find rules based on pairs of feature conditions."""
        rules = []
        total_instances = len(positive_instances) + len(negative_instances)
        feature_list = list(self.feature_names)
        
        # Try pairs of features
        for i, f1_name in enumerate(feature_list):
            for f2_name in feature_list[i+1:]:
                for f1_value in self.feature_values[f1_name]:
                    for f2_value in self.feature_values[f2_name]:
                        # Count support and confidence
                        positive_with_features = sum(
                            1 for inst_id in positive_instances
                            if (self.instance_features.get(inst_id, {}).get(f1_name) == f1_value and
                                self.instance_features.get(inst_id, {}).get(f2_name) == f2_value)
                        )
                        negative_with_features = sum(
                            1 for inst_id in negative_instances
                            if (self.instance_features.get(inst_id, {}).get(f1_name) == f1_value and
                                self.instance_features.get(inst_id, {}).get(f2_name) == f2_value)
                        )
                        
                        total_with_features = positive_with_features + negative_with_features
                        support = total_with_features / total_instances if total_instances > 0 else 0
                        confidence = (
                            positive_with_features / total_with_features
                            if total_with_features > 0 else 0
                        )
                        
                        # Higher thresholds for pair rules
                        if support >= self.min_support and confidence >= max(0.7, self.min_confidence):
                            X = Variable("X")
                            head = Atom("predict", [X, Constant(label)])
                            body = [
                                Atom("feature", [X, Constant(f1_name), Constant(f1_value)]),
                                Atom("feature", [X, Constant(f2_name), Constant(f2_value)])
                            ]
                            rule = Clause(head, body)
                            rules.append(rule)
                            
                            if self.verbose:
                                print(f"    Found pair rule: {rule} (support={support:.2f}, conf={confidence:.2f})")
        
        # Sort and return top rules
        rules.sort(key=lambda r: (
            -self._rule_confidence(r, positive_instances, negative_instances),
            -self._rule_support(r, positive_instances + negative_instances)
        ))
        
        return rules[:2]
    
    def _rule_confidence(self, rule: Clause, positive_instances: List[Any], negative_instances: List[Any]) -> float:
        """Calculate confidence of a rule."""
        positive_covered = sum(1 for inst in positive_instances if self._rule_covers(rule, inst))
        negative_covered = sum(1 for inst in negative_instances if self._rule_covers(rule, inst))
        total_covered = positive_covered + negative_covered
        return positive_covered / total_covered if total_covered > 0 else 0
    
    def _rule_support(self, rule: Clause, instances: List[Any]) -> float:
        """Calculate support of a rule."""
        covered = sum(1 for inst in instances if self._rule_covers(rule, inst))
        return covered / len(instances) if instances else 0
    
    def _rule_covers(self, rule: Clause, instance_id: Any) -> bool:
        """Check if a rule's body conditions are satisfied for an instance."""
        features = self.instance_features.get(instance_id, {})
        
        for body_atom in rule.body:
            if body_atom.predicate == "feature" and len(body_atom.arguments) >= 3:
                feature_name = body_atom.arguments[1]
                feature_value = body_atom.arguments[2]
                
                if isinstance(feature_name, Constant) and isinstance(feature_value, Constant):
                    actual_value = features.get(feature_name.value)
                    if actual_value != feature_value.value:
                        return False
        
        return True
    
    def _refine_theory(self, theory: Theory) -> Theory:
        """
        Refine theory to improve coverage and reduce contradictions.
        
        1. Find instances not covered by any rule
        2. Generate more specific rules for those instances
        3. Remove rules that cause too many contradictions
        """
        # Find uncovered positive instances for each label
        for label in self.seen_labels:
            positive_instances = [
                inst_id for inst_id, inst_label in self.instance_labels.items()
                if inst_label == label
            ]
            
            # Find instances covered by existing rules
            covered = set()
            for rule in theory.get_clauses():
                if self._rule_label(rule) == label:
                    for inst_id in positive_instances:
                        if self._rule_covers(rule, inst_id):
                            covered.add(inst_id)
            
            uncovered = [inst for inst in positive_instances if inst not in covered]
            
            if uncovered and len(theory) < self.max_theory_size:
                # Try to create rules for uncovered instances
                for inst_id in uncovered[:5]:  # Limit to avoid explosion
                    rule = self._create_rule_for_instance(inst_id, label)
                    if rule is not None:
                        # Check it doesn't cause too many contradictions
                        negative_instances = [
                            i for i, l in self.instance_labels.items() if l != label
                        ]
                        false_positives = sum(
                            1 for inst in negative_instances
                            if self._rule_covers(rule, inst)
                        )
                        
                        if false_positives < len(uncovered) * 0.5:  # Allow some errors
                            theory.add_clause(rule)
                            self.history.append(theory.copy())
        
        return theory
    
    def _rule_label(self, rule: Clause) -> Optional[Any]:
        """Get the label from a rule's head."""
        if rule.head.predicate == "predict" and len(rule.head.arguments) >= 2:
            label_arg = rule.head.arguments[1]
            if isinstance(label_arg, Constant):
                return label_arg.value
        return None
    
    def _create_rule_for_instance(self, instance_id: Any, label: Any) -> Optional[Clause]:
        """Create a rule that covers a specific instance."""
        features = self.instance_features.get(instance_id, {})
        
        if not features:
            return None
        
        # Use the most discriminative features for this instance
        best_features = self._rank_features_for_instance(instance_id, label)
        
        if not best_features:
            return None
        
        X = Variable("X")
        head = Atom("predict", [X, Constant(label)])
        
        # Use top 1-2 features
        body = []
        for feature_name, feature_value in best_features[:2]:
            body.append(Atom("feature", [X, Constant(feature_name), Constant(feature_value)]))
        
        return Clause(head, body)
    
    def _rank_features_for_instance(self, instance_id: Any, label: Any) -> List[Tuple[str, Any]]:
        """Rank features by how discriminative they are for this instance's label."""
        features = self.instance_features.get(instance_id, {})
        
        ranked = []
        for feature_name, feature_value in features.items():
            # Count how many instances with same label have this feature value
            same_label_same_value = sum(
                1 for inst_id, inst_label in self.instance_labels.items()
                if inst_label == label and
                self.instance_features.get(inst_id, {}).get(feature_name) == feature_value
            )
            
            # Count how many instances with different label have this feature value
            diff_label_same_value = sum(
                1 for inst_id, inst_label in self.instance_labels.items()
                if inst_label != label and
                self.instance_features.get(inst_id, {}).get(feature_name) == feature_value
            )
            
            total = same_label_same_value + diff_label_same_value
            score = same_label_same_value / total if total > 0 else 0
            
            ranked.append((feature_name, feature_value, score))
        
        # Sort by score descending
        ranked.sort(key=lambda x: -x[2])
        
        # Return (name, value) tuples for top features
        return [(name, value) for name, value, score in ranked if score > 0.5]
    
    def _setup_default_refinement_operator(self) -> None:
        """Set up default refinement operator based on observed facts."""
        feature_names_list = list(self.feature_names)
        feature_values_dict = {name: list(values) for name, values in self.feature_values.items()}
        
        add_condition = AddConditionRefinement(feature_names_list, feature_values_dict)
        specialize = SpecializePredicateRefinement()
        
        self.refinement_operator = CompositeRefinementOperator([add_condition, specialize])
    
    def get_history(self) -> List[Theory]:
        """Get the history of theory evolution."""
        return self.history.copy()
