"""
CRS Memory Library - Builder
High-level API for constructing MemoryStructures.
"""
from typing import Optional
import datetime
import uuid

from .structure import MemoryStructure, MemoryBlock, MemoryCell, Meta, BlockType, CellRole
from .ast_logic import ASTNormalizer
from .causal import CausalExtractor
from .io import MemoryIO

class MemoryBuilder:
    """
    Builder facade for creating CRS Memory objects.
    """
    
    def __init__(self, spec_dim: int = 1024):
        self.spec_dim = spec_dim
        self.normalizer = ASTNormalizer()
        self.extractor = CausalExtractor()

    def from_formula(self, source: str, fmt: str = "sympy", structure_id: str = None, block_id: str = None) -> MemoryStructure:
        """
        Create a MemoryStructure from a formula string.
        """
        if not structure_id:
            structure_id = f"S-{uuid.uuid4().hex[:8]}"
        if not block_id:
            block_id = f"B-{uuid.uuid4().hex[:8]}"
            
        # 1. Parse & Normalize
        if fmt != "sympy":
             raise NotImplementedError("Only sympy format supported in v0.1")
        
        expr = self.normalizer.parse_formula(source)
        norm_ast = self.normalizer.normalize(expr)
        
        # 2. Extract Causal Model
        causal_graph = self.extractor.extract(norm_ast)
        
        # 3. Construct Block
        block = MemoryBlock(
            id=block_id,
            block_type=BlockType.FORMULA,
            cells=[], # No atoms generated in this high-level pass yet
            payload={
                "source": source,
                "fmt": fmt,
                "canon_ast": norm_ast.canon_ast,
                "normalized_signature": norm_ast.signature,
                "var_map": norm_ast.var_map
            },
            causal_model=causal_graph
        )
        
        # 4. Construct Structure
        meta = Meta(
            id=structure_id,
            created_at=datetime.datetime.now().isoformat(),
            source="MemoryBuilder.from_formula"
        )
        
        structure = MemoryStructure(
            schema_version="0.1",
            meta=meta,
            blocks=[block],
            causal_graph=causal_graph # Lift block causal graph to structure level for v0.1 simple case
        )
        
        return structure

    def load_structure(self, yaml_content: str) -> MemoryStructure:
        return MemoryIO.from_yaml(yaml_content)

    def save_structure(self, structure: MemoryStructure) -> str:
        return MemoryIO.to_yaml(structure)
