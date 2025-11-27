"""Unit tests for resolution engine."""

import unittest
from src.core.atoms import Atom, Variable, Constant
from src.core.clauses import Clause
from src.core.theory import Theory
from src.core.resolution import ResolutionEngine


class TestResolution(unittest.TestCase):
    """Test cases for SLD resolution."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = ResolutionEngine(max_depth=10)
    
    def test_prove_fact(self):
        """Test proving a fact directly."""
        theory = Theory()
        fact = Clause(Atom("p", [Constant(1)]), [])
        theory.add_clause(fact)
        
        goal = Atom("p", [Constant(1)])
        proof = self.engine.prove(theory, goal)
        self.assertIsNotNone(proof)
        self.assertTrue(self.engine.can_prove(theory, goal))
    
    def test_prove_rule(self):
        """Test proving via a rule."""
        theory = Theory()
        # Rule: p(X) :- q(X)
        rule = Clause(
            Atom("p", [Variable("X")]),
            [Atom("q", [Variable("X")])]
        )
        theory.add_clause(rule)
        
        # Fact: q(1)
        fact = Clause(Atom("q", [Constant(1)]), [])
        theory.add_clause(fact)
        
        # Should be able to prove p(1)
        goal = Atom("p", [Constant(1)])
        self.assertTrue(self.engine.can_prove(theory, goal))
    
    def test_cannot_prove(self):
        """Test that unprovable goals return None."""
        theory = Theory()
        fact = Clause(Atom("p", [Constant(1)]), [])
        theory.add_clause(fact)
        
        goal = Atom("p", [Constant(2)])
        proof = self.engine.prove(theory, goal)
        # Should not be able to prove
        self.assertFalse(self.engine.can_prove(theory, goal))
    
    def test_find_contradiction(self):
        """Test finding contradictions."""
        theory = Theory()
        # Theory says p(1) is true
        fact = Clause(Atom("p", [Constant(1)]), [])
        theory.add_clause(fact)
        
        # Oracle says p(1) is false
        def false_oracle(atom):
            return False
        
        goal = Atom("p", [Constant(1)])
        proof = self.engine.find_contradiction(theory, goal, false_oracle)
        self.assertIsNotNone(proof)


if __name__ == "__main__":
    unittest.main()


