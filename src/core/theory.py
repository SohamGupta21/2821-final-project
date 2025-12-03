"""Theory management - collection of clauses."""

from typing import List, Set, Optional
from .clauses import Clause
from .atoms import Atom


class Theory:
    """A theory is a collection of Horn clauses."""
    
    def __init__(self, clauses: Optional[List[Clause]] = None):
        """Initialize a theory with optional list of clauses."""
        self._clauses: List[Clause] = clauses.copy() if clauses else []
        self._clause_set: Set[Clause] = set(self._clauses)
    
    def add_clause(self, clause: Clause) -> None:
        """Add a clause to the theory."""
        if clause not in self._clause_set:
            self._clauses.append(clause)
            self._clause_set.add(clause)
    
    def remove_clause(self, clause: Clause) -> bool:
        """Remove a clause from the theory. Returns True if removed, False if not found."""
        if clause in self._clause_set:
            self._clauses.remove(clause)
            self._clause_set.remove(clause)
            return True
        return False
    
    def get_clauses(self) -> List[Clause]:
        """Get all clauses in the theory."""
        return self._clauses.copy()
    
    def get_facts(self) -> List[Clause]:
        """Get all fact clauses (clauses with no body)."""
        return [clause for clause in self._clauses if clause.is_fact()]
    
    def get_rules(self) -> List[Clause]:
        """Get all rule clauses (clauses with body)."""
        return [clause for clause in self._clauses if clause.is_rule()]
    
    def __len__(self) -> int:
        """Return the number of clauses in the theory."""
        return len(self._clauses)
    
    def __contains__(self, clause: Clause) -> bool:
        """Check if a clause is in the theory."""
        return clause in self._clause_set
    
    def __iter__(self):
        """Iterate over clauses in the theory."""
        return iter(self._clauses)
    
    def __str__(self) -> str:
        """String representation of the theory."""
        if not self._clauses:
            return "Theory([])"
        clauses_str = "\n".join(f"  {clause}" for clause in self._clauses)
        return f"Theory([\n{clauses_str}\n])"
    
    def __repr__(self) -> str:
        return f"Theory({repr(self._clauses)})"
    
    def copy(self) -> 'Theory':
        """Create a copy of this theory."""
        return Theory(self._clauses.copy())
    
    def clear(self) -> None:
        """Remove all clauses from the theory."""
        self._clauses.clear()
        self._clause_set.clear()
    
    def find_clauses_with_head(self, atom: Atom) -> List[Clause]:
        """Find all clauses whose head unifies with the given atom."""
        from .atoms import unify_atoms
        matching = []
        for clause in self._clauses:
            if unify_atoms(clause.head, atom) is not None:
                matching.append(clause)
        return matching

