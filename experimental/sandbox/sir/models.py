"""
SIR v1.0 Data Models
Implements the Official SIR Schema with Structure Signature calculation.
"""

import hashlib
import json
from typing import List, Dict, Optional, Literal, Any, Union
from pydantic import BaseModel, Field, ConfigDict

# --- Hashing Utilities ---
def stable_hash(data: Any) -> str:
    """Computes a stable SHA-256 hash for a dictionary or string."""
    if isinstance(data, dict):
        # Sort keys for stability
        serialized = json.dumps(data, sort_keys=True)
    else:
        serialized = str(data)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

# --- Semantic Core Components ---

class EntityAttributes(BaseModel):
    quantifier: Literal["forall", "exists", "none"] = "none"
    domain: Literal["number", "boolean", "sequence", "abstract"] = "abstract"
    role: Literal["subject", "object", "operand", "iterator"] = "subject"

class Entity(BaseModel):
    id: str
    type: Literal["concept", "variable", "constant", "function"]
    label: str  # Surface form (ignored in hash)
    attributes: EntityAttributes

    def get_structure_feature(self) -> str:
        """
        phi(E): Structural feature of the entity.
        Ignores 'label' and 'id' (unless id carries structural meaning, usually just a handle).
        Currently: Hash(type, attributes)
        """
        data = {
            "type": self.type,
            "attributes": self.attributes.model_dump()
        }
        return stable_hash(data)

class Relation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: str
    type: Literal["comparison", "causal", "hierarchical", "dataflow"]
    from_id: str = Field(alias="from")
    to_id: str = Field(alias="to")
    polarity: Literal["positive", "negative", "neutral"] = "neutral"

    def get_structure_feature(self, entity_map: Dict[str, str]) -> str:
        """
        psi(R): Structural feature of the relation.
        Includes hashes of connected entities to capture topology (1-hop).
        entity_map: id -> entity_feature
        """
        # We assume entity_map provides the structural feature of the endpoints.
        # This makes the relation hash dependent on the *types* of nodes it connects vs just IDs.
        src_hash = entity_map.get(self.from_id, "unknown")
        dst_hash = entity_map.get(self.to_id, "unknown")
        
        data = {
            "type": self.type,
            "polarity": self.polarity,
            "src_structure": src_hash, 
            "dst_structure": dst_hash
        }
        return stable_hash(data)

class OperationProperties(BaseModel):
    commutative: bool = False
    associative: bool = False
    side_effect: bool = False

class Operation(BaseModel):
    id: str
    operator: Literal["add", "multiply", "compare", "assign", "loop", "map"]
    operands: List[str] # List of Entity IDs
    properties: OperationProperties

    def get_structure_feature(self, entity_map: Dict[str, str]) -> str:
        """
        omega(O): Structural feature of the operation.
        """
        # Resolve operands to their structural features
        operand_hashes = [entity_map.get(op_id, "unknown") for op_id in self.operands]
        
        # If commutative, sort operands to ensure 3+x == x+3
        if self.properties.commutative:
            operand_hashes.sort()
            
        data = {
            "operator": self.operator,
            "properties": self.properties.model_dump(),
            "operands_structure": operand_hashes
        }
        return stable_hash(data)

class Constraint(BaseModel):
    id: str
    type: Literal["logical", "numerical", "boundary"]
    expression: str # Normalized expression string
    scope: List[str] # List of Entity IDs

    def get_structure_feature(self, entity_map: Dict[str, str]) -> str:
        scope_hashes = sorted([entity_map.get(s, "unknown") for s in self.scope])
        data = {
            "type": self.type,
            "expression": self.expression, # Expression itself is structural? 
            # If expression contains labels like "x > 5", it violates abstraction if labels strictly matter.
            # But "normalized_constraint" implies variable names are normalized (e.g. $1 > $2).
            # For v1.0, we assume expression is normalized.
            "scope_structure": scope_hashes
        }
        return stable_hash(data)

class SemanticCore(BaseModel):
    entities: List[Entity] = []
    relations: List[Relation] = []
    operations: List[Operation] = []
    constraints: List[Constraint] = []

class StructureSignature(BaseModel):
    graph_hash: str
    depth: int
    branching_factor: float

class SIR(BaseModel):
    sir_version: str = "1.0"
    modality: Literal["natural_language", "math", "code"]
    semantic_core: SemanticCore
    structure_signature: StructureSignature
    abstraction_level: float
    confidence: float

    def recompute_signature(self):
        """
        Calculates and updates the StructureSignature based on current SemanticCore.
        """
        # 1. Calculate Entity Features (phi)
        # Map ID -> Feature
        entity_map = {e.id: e.get_structure_feature() for e in self.semantic_core.entities}
        
        # 2. Collect all features
        features = []
        features.extend(entity_map.values())
        features.extend([r.get_structure_feature(entity_map) for r in self.semantic_core.relations])
        features.extend([o.get_structure_feature(entity_map) for o in self.semantic_core.operations])
        features.extend([c.get_structure_feature(entity_map) for c in self.semantic_core.constraints])
        
        # 3. Aggregation (Order Independent)
        # Using sorted concatenation for hash stability
        features.sort()
        combined = "".join(features)
        graph_hash = hashlib.sha256(combined.encode('utf-8')).hexdigest()
        
        # 4. Metrics (Simple approximation for v1.0)
        depth = 0
        # Simple depth heuristic: max chain length of relations? 
        # For now, placeholder.
        
        branching = 0.0
        if self.semantic_core.entities:
            # Avg relations per entity
            branching = len(self.semantic_core.relations) / len(self.semantic_core.entities)

        self.structure_signature = StructureSignature(
            graph_hash=graph_hash,
            depth=depth,
            branching_factor=branching
        )
