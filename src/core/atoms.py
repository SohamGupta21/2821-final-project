"""Atom representation and unification algorithm."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
from abc import ABC, abstractmethod


class Term(ABC):
    """Base class for terms (variables, constants, structured terms)."""
    
    @abstractmethod
    def __str__(self) -> str:
        pass
    
    @abstractmethod
    def __eq__(self, other) -> bool:
        pass
    
    @abstractmethod
    def __hash__(self) -> int:
        pass
    
    @abstractmethod
    def apply_substitution(self, substitution: Dict['Variable', 'Term']) -> 'Term':
        """Apply a substitution to this term."""
        pass
    
    @abstractmethod
    def is_variable(self) -> bool:
        """Check if this term is a variable."""
        pass


@dataclass(frozen=True)
class Variable(Term):
    """A logical variable."""
    name: str
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return f"Variable('{self.name}')"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Variable):
            return False
        return self.name == other.name
    
    def __hash__(self) -> int:
        return hash(('Variable', self.name))
    
    def apply_substitution(self, substitution: Dict['Variable', 'Term']) -> 'Term':
        return substitution.get(self, self)
    
    def is_variable(self) -> bool:
        return True


@dataclass(frozen=True)
class Constant(Term):
    """A constant term."""
    value: Any
    
    def __str__(self) -> str:
        return str(self.value)
    
    def __repr__(self) -> str:
        return f"Constant({repr(self.value)})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Constant):
            return False
        return self.value == other.value
    
    def __hash__(self) -> int:
        return hash(('Constant', self.value))
    
    def apply_substitution(self, substitution: Dict[Variable, Term]) -> 'Term':
        return self
    
    def is_variable(self) -> bool:
        return False


@dataclass(frozen=True)
class Atom:
    """A logical atom (predicate with arguments)."""
    predicate: str
    arguments: List[Term]
    
    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self.arguments)
        return f"{self.predicate}({args_str})"
    
    def __repr__(self) -> str:
        args_repr = ", ".join(repr(arg) for arg in self.arguments)
        return f"Atom('{self.predicate}', [{args_repr}])"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Atom):
            return False
        return (self.predicate == other.predicate and 
                len(self.arguments) == len(other.arguments) and
                all(a == b for a, b in zip(self.arguments, other.arguments)))
    
    def __hash__(self) -> int:
        return hash((self.predicate, tuple(self.arguments)))
    
    def is_ground(self) -> bool:
        """Check if this atom is ground (contains no variables)."""
        return all(not arg.is_variable() for arg in self.arguments)
    
    def apply_substitution(self, substitution: Dict[Variable, Term]) -> 'Atom':
        """Apply a substitution to all terms in this atom."""
        new_args = [arg.apply_substitution(substitution) for arg in self.arguments]
        return Atom(self.predicate, new_args)
    
    def get_variables(self) -> List[Variable]:
        """Get all variables in this atom."""
        variables = []
        for arg in self.arguments:
            if isinstance(arg, Variable):
                variables.append(arg)
        return variables


def unify(term1: Term, term2: Term, substitution: Optional[Dict[Variable, Term]] = None) -> Optional[Dict[Variable, Term]]:
    """
    Unify two terms, returning a substitution that makes them equal.
    
    Returns None if unification is impossible.
    """
    if substitution is None:
        substitution = {}
    
    # If terms are already equal, return current substitution
    if term1 == term2:
        return substitution
    
    # If term1 is a variable
    if isinstance(term1, Variable):
        # Check for occurs check
        if term1 in substitution:
            return unify(substitution[term1], term2, substitution)
        # Check if term2 contains term1 (occurs check)
        if isinstance(term2, Variable) and term2 in substitution:
            return unify(term1, substitution[term2], substitution)
        # Add binding
        new_sub = substitution.copy()
        new_sub[term1] = term2
        return new_sub
    
    # If term2 is a variable
    if isinstance(term2, Variable):
        return unify(term2, term1, substitution)
    
    # Both are constants - must be equal
    if isinstance(term1, Constant) and isinstance(term2, Constant):
        if term1.value == term2.value:
            return substitution
        return None
    
    # Cannot unify different types
    return None


def unify_atoms(atom1: Atom, atom2: Atom) -> Optional[Dict[Variable, Term]]:
    """
    Unify two atoms, returning a substitution that makes them equal.
    
    Returns None if unification is impossible.
    """
    if atom1.predicate != atom2.predicate:
        return None
    
    if len(atom1.arguments) != len(atom2.arguments):
        return None
    
    substitution = {}
    for arg1, arg2 in zip(atom1.arguments, atom2.arguments):
        substitution = unify(arg1, arg2, substitution)
        if substitution is None:
            return None
    
    return substitution


def compose_substitutions(sub1: Dict[Variable, Term], sub2: Dict[Variable, Term]) -> Dict[Variable, Term]:
    """
    Compose two substitutions: sub1 o sub2.
    
    First apply sub2, then apply sub1 to the result.
    """
    # Apply sub1 to all terms in sub2
    composed = {}
    for var, term in sub2.items():
        composed[var] = term.apply_substitution(sub1)
    
    # Add bindings from sub1 that aren't in sub2
    for var, term in sub1.items():
        if var not in composed:
            composed[var] = term.apply_substitution(sub2)
    
    return composed


