"""Horn clause representation."""

from dataclasses import dataclass
from typing import List, Dict, Optional
from .atoms import Atom, Variable, Term


@dataclass
class Clause:
    """A Horn clause: head :- body."""
    head: Atom
    body: List[Atom]
    
    def __str__(self) -> str:
        if not self.body:
            return str(self.head) + "."
        body_str = ", ".join(str(atom) for atom in self.body)
        return f"{self.head} :- {body_str}."
    
    def __repr__(self) -> str:
        return f"Clause({repr(self.head)}, {repr(self.body)})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Clause):
            return False
        return self.head == other.head and self.body == other.body
    
    def __hash__(self) -> int:
        return hash((self.head, tuple(self.body)))
    
    def is_fact(self) -> bool:
        """Check if this clause is a fact (no body)."""
        return len(self.body) == 0
    
    def is_rule(self) -> bool:
        """Check if this clause is a rule (has body)."""
        return len(self.body) > 0
    
    def apply_substitution(self, substitution: Dict[Variable, Term]) -> 'Clause':
        """Apply a substitution to all atoms in this clause."""
        new_head = self.head.apply_substitution(substitution)
        new_body = [atom.apply_substitution(substitution) for atom in self.body]
        return Clause(new_head, new_body)
    
    def get_variables(self) -> List[Variable]:
        """Get all variables in this clause."""
        variables = []
        variables.extend(self.head.get_variables())
        for atom in self.body:
            variables.extend(atom.get_variables())
        # Remove duplicates while preserving order
        seen = set()
        unique_vars = []
        for var in variables:
            if var not in seen:
                seen.add(var)
                unique_vars.append(var)
        return unique_vars
    
    def is_ground(self) -> bool:
        """Check if this clause is ground (contains no variables)."""
        return self.head.is_ground() and all(atom.is_ground() for atom in self.body)
    
    def standardize_apart(self, used_variables: Optional[List[Variable]] = None) -> 'Clause':
        """
        Standardize apart variables in this clause to avoid naming conflicts.
        
        Returns a new clause with renamed variables.
        """
        if used_variables is None:
            used_variables = []
        
        variables = self.get_variables()
        substitution = {}
        counter = 0
        
        for var in variables:
            if var in used_variables:
                # Create a new variable name
                new_name = f"{var.name}_{counter}"
                counter += 1
                new_var = Variable(new_name)
                substitution[var] = new_var
                used_variables.append(new_var)
            else:
                used_variables.append(var)
        
        if substitution:
            return self.apply_substitution(substitution)
        return Clause(self.head, self.body.copy())
    
    def pretty_print(self, indent: int = 0) -> str:
        """Pretty print the clause with indentation."""
        indent_str = " " * indent
        if not self.body:
            return indent_str + str(self.head) + "."
        
        head_str = str(self.head)
        body_str = ",\n" + indent_str + "    ".join(str(atom) for atom in self.body)
        return indent_str + head_str + " :-\n" + indent_str + "    " + body_str + "."

