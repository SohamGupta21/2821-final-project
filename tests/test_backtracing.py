"""Unit tests for backtracing."""

import unittest
from src.core.atoms import Atom, Constant
from src.core.clauses import Clause
from src.core.theory import Theory
from src.inference.backtracing import ContradictionBacktracer
from src.inference.oracle import Oracle


class MockOracle(Oracle):
    """Mock oracle for testing."""
    
    def __init__(self, truth_values):
        self.truth_values = truth_values
    
    def query(self, atom):
        return self.truth_values.get(str(atom), False)


class TestBacktracing(unittest.TestCase):
    """Test cases for contradiction backtracing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.backtracer = ContradictionBacktracer()
    
    def test_find_contradiction(self):
        """Test finding a contradiction."""
        theory = Theory()
        # Theory has a false fact
        false_fact = Clause(Atom("p", [Constant(1)]), [])
        theory.add_clause(false_fact)
        
        # Oracle says p(1) is false
        oracle = MockOracle({str(Atom("p", [Constant(1)])): False})
        
        fact = Atom("p", [Constant(1)])
        false_clause = self.backtracer.find_contradiction(theory, fact, oracle)
        self.assertIsNotNone(false_clause)
        self.assertEqual(false_clause, false_fact)
    
    def test_no_contradiction(self):
        """Test when there's no contradiction."""
        theory = Theory()
        fact_clause = Clause(Atom("p", [Constant(1)]), [])
        theory.add_clause(fact_clause)
        
        # Oracle says p(1) is true
        oracle = MockOracle({str(Atom("p", [Constant(1)])): True})
        
        fact = Atom("p", [Constant(1)])
        false_clause = self.backtracer.find_contradiction(theory, fact, oracle)
        self.assertIsNone(false_clause)
    
    def test_is_too_strong(self):
        """Test checking if theory is too strong."""
        theory = Theory()
        false_fact = Clause(Atom("p", [Constant(1)]), [])
        theory.add_clause(false_fact)
        
        oracle = MockOracle({str(Atom("p", [Constant(1)])): False})
        facts = [Atom("p", [Constant(1)])]
        
        self.assertTrue(self.backtracer.is_too_strong(theory, facts, oracle))


if __name__ == "__main__":
    unittest.main()


