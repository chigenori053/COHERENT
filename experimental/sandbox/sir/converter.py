"""
SIR Converter / Factory
Helper utilities to generate SIR instances from various inputs.
"""

import uuid
from typing import Dict, Any, List, Optional
from .models import (
    SIR, SemanticCore, StructureSignature,
    Entity, Relation, Operation, Constraint,
    EntityAttributes, OperationProperties
)

class SIRFactory:
    """
    Factory class to construct SIR objects easily.
    """
    
    @staticmethod
    def create_empty(modality: str = "math") -> SIR:
        return SIR(
            modality=modality,
            semantic_core=SemanticCore(),
            structure_signature=StructureSignature(graph_hash="", depth=0, branching_factor=0.0),
            abstraction_level=0.0,
            confidence=1.0
        )

    @staticmethod
    def from_math_expression(expr: str) -> SIR:
        """
        Naive parser for simple math expressions (Sandbox implementation).
        Supports:
        - "a + b" (Operation: add)
        - "x > y" (Relation: comparison)
        - "x = y" (Relation: comparison)
        """
        sir = SIRFactory.create_empty("math")
        core = sir.semantic_core
        
        # Simple tokenization by spaces
        tokens = expr.split()
        
        # Heuristic parsing
        if ">" in tokens:
            op_idx = tokens.index(">")
            lhs, rhs = tokens[:op_idx], tokens[op_idx+1:]
            
            # Create Entities
            e1 = SIRFactory._create_entity(core, "".join(lhs), "variable")
            e2 = SIRFactory._create_entity(core, "".join(rhs), "variable")
            
            # Create Relation
            rel = Relation(
                id=f"R_{uuid.uuid4().hex[:8]}",
                type="comparison",
                # "from_id" aliased to "from", "to_id" aliased to "to".
                # ConfigDict(populate_by_name=True) allows using field names.
                from_id=e1.id, 
                to_id=e2.id,
                polarity="positive"
            )
            core.relations.append(rel)
            
        elif "+" in tokens:
            op_idx = tokens.index("+")
            lhs, rhs = tokens[:op_idx], tokens[op_idx+1:]
            
            e1 = SIRFactory._create_entity(core, "".join(lhs), "variable")
            e2 = SIRFactory._create_entity(core, "".join(rhs), "variable")
            
            op = Operation(
                id=f"O_{uuid.uuid4().hex[:8]}",
                operator="add",
                operands=[e1.id, e2.id],
                properties=OperationProperties(commutative=True)
            )
            core.operations.append(op)
            
        else:
            # Atomic entity?
            SIRFactory._create_entity(core, expr, "variable")
            
        # Finalize structure
        sir.recompute_signature()
        # Abstraction level calc (Naive)
        sir.abstraction_level = 0.5 
        return sir

    @staticmethod
    def _create_entity(core: SemanticCore, label: str, type_: str) -> Entity:
        # Check if entity already exists? For now, create new.
        # In a real parser, we'd resolve scope.
        # For "3+x vs x+3", if we parse separately, IDs are different.
        # But GraphHash depends on Type+Attrs.
        # So even if IDs differ, if Type/Attributes are same, result is same.
        
        # Detect constant if numeric
        if label.isdigit():
            type_ = "constant"
            attrs = EntityAttributes(domain="number", role="operand")
        else:
            attrs = EntityAttributes(domain="abstract", role="operand")

        ent = Entity(
            id=f"E_{uuid.uuid4().hex[:8]}",
            type=type_, # type: ignore (Literals are strings at runtime)
            label=label,
            attributes=attrs
        )
        core.entities.append(ent)
        return ent
