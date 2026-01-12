"""
Semantic Intermediate Representation (SIR) Core Package.
Provides unified data structures and projection logic for multi-modal meaning.
"""

from .models import SIR, Entity, Relation, Operation, Constraint, EntityAttributes
from .converter import SIRFactory
from .projection import SIRProjector

__all__ = [
    "SIR",
    "Entity",
    "Relation",
    "Operation",
    "Constraint",
    "EntityAttributes",
    "SIRFactory",
    "SIRProjector"
]
