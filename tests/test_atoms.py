"""Unit tests for atoms module."""

import unittest
from src.core.atoms import Atom, Variable, Constant, unify, unify_atoms


class TestAtoms(unittest.TestCase):
    """Test cases for Atom, Term, and unification."""
    
    def test_variable_creation(self):
        """Test variable creation."""
        x = Variable("X")
        self.assertEqual(str(x), "X")
        self.assertTrue(x.is_variable())
    
    def test_constant_creation(self):
        """Test constant creation."""
        c = Constant(5)
        self.assertEqual(str(c), "5")
        self.assertFalse(c.is_variable())
    
    def test_atom_creation(self):
        """Test atom creation."""
        atom = Atom("pred", [Variable("X"), Constant(1)])
        self.assertEqual(atom.predicate, "pred")
        self.assertEqual(len(atom.arguments), 2)
        self.assertFalse(atom.is_ground())
    
    def test_ground_atom(self):
        """Test ground atom detection."""
        atom = Atom("pred", [Constant(1), Constant(2)])
        self.assertTrue(atom.is_ground())
    
    def test_unify_variables(self):
        """Test unification of variables."""
        x = Variable("X")
        y = Variable("Y")
        result = unify(x, y)
        self.assertIsNotNone(result)
        self.assertIn(x, result)
    
    def test_unify_variable_constant(self):
        """Test unification of variable and constant."""
        x = Variable("X")
        c = Constant(5)
        result = unify(x, c)
        self.assertIsNotNone(result)
        self.assertEqual(result[x], c)
    
    def test_unify_constants(self):
        """Test unification of constants."""
        c1 = Constant(5)
        c2 = Constant(5)
        result = unify(c1, c2)
        self.assertIsNotNone(result)
        
        c3 = Constant(6)
        result2 = unify(c1, c3)
        self.assertIsNone(result2)
    
    def test_unify_atoms(self):
        """Test atom unification."""
        atom1 = Atom("pred", [Variable("X"), Constant(1)])
        atom2 = Atom("pred", [Constant(2), Constant(1)])
        result = unify_atoms(atom1, atom2)
        self.assertIsNotNone(result)
        self.assertEqual(result[Variable("X")], Constant(2))
    
    def test_unify_atoms_different_predicates(self):
        """Test that different predicates don't unify."""
        atom1 = Atom("pred1", [Variable("X")])
        atom2 = Atom("pred2", [Variable("X")])
        result = unify_atoms(atom1, atom2)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()

