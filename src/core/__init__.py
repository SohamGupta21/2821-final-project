"""Core data structures for logical reasoning."""

from .atoms import Atom, Term, Variable, Constant
from .clauses import Clause
from .theory import Theory
from .resolution import ResolutionEngine, ProofNode

__all__ = [
    "Atom",
    "Term",
    "Variable",
    "Constant",
    "Clause",
    "Theory",
    "ResolutionEngine",
    "ProofNode",
]


