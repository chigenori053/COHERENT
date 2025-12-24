"""
CRS Memory Library - Structure
Hierarchical composition of MemoryAtoms into Cells, Blocks, and Structures.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Literal
from enum import Enum
import datetime

from .atoms import MemoryAtom

@dataclass
class AtomRef:
    """Reference to a MemoryAtom."""
    atom_id: str
    # Could imply embedding or just ID ref. v0.1 spec says ID ref.

class CellRole(Enum):
    CAUSE = "cause"
    EFFECT = "effect"
    CONTEXT = "context"
    EVIDENCE = "evidence"
    NOTE = "note"

@dataclass
class MemoryCell:
    """Group of atoms with a causal role."""
    id: str
    atoms: List[AtomRef]
    role: CellRole
    local_confidence: float = 1.0

class BlockType(Enum):
    NUMERIC = "numeric"
    TEXT = "text"
    FORMULA = "formula"
    LIST = "list"
    TABLE = "table"
    MIXED = "mixed"

@dataclass
class BlockLink:
    target_block_id: str
    rel: str # derived_from, supports, contradicts, ...

@dataclass
class Meta:
    id: str
    created_at: str
    source: str = ""
    title: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)

@dataclass
class CausalVariable:
    id: str
    name: str
    vtype: str = "numeric" # numeric, categorical, text, latent
    source: str = "ast_input" # ast_input, ast_derived, ast_output

@dataclass
class CausalRelation:
    from_id: str
    to_id: str
    strength: float = 0.5
    confidence: float = 0.8
    mechanism: str = "unknown"

@dataclass
class CausalGraph:
    nodes: Dict[str, CausalVariable] # id -> Variable
    edges: List[CausalRelation]

@dataclass
class MemoryBlock:
    """Semantic unit representing numeric/text/formula/list/table/mixed."""
    id: str
    block_type: BlockType
    cells: List[MemoryCell]
    payload: Dict[str, Any] # Must be JSON/YAML safe
    causal_model: Optional[CausalGraph] = None
    links: List[BlockLink] = field(default_factory=list)

@dataclass
class MemoryStructure:
    """External interchange unit; fully serializable."""
    schema_version: str # "0.1"
    meta: Meta
    blocks: List[MemoryBlock]
    causal_graph: Optional[CausalGraph] = None # Global causal graph optionally merged
    
    # Optional dictionaries for self-contained portability
    atoms: Dict[str, Any] = field(default_factory=dict) # id -> atom dict or obj
    exports: Dict[str, Any] = field(default_factory=dict)
