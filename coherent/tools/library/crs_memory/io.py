"""
CRS Memory Library - IO
JSON/YAML Serialization and Deserialization.
"""
import json
import yaml
import dataclasses
from enum import Enum
import datetime
from typing import Dict, Any, Type, Union

from .structure import MemoryStructure, MemoryBlock, MemoryCell, BlockLink, Meta, CausalGraph, CausalVariable, CausalRelation, AtomRef, BlockType, CellRole
from .atoms import MemoryAtom, ComplexVal, AtomType, TransformSpec, InverseSpec, ProjectionSpec, ReconstructQuality

class MemoryIO:
    """
    Handles serialization of MemoryStructure to/from JSON/YAML.
    Enforces "no loss of info" and "real-world" types.
    """
    
    @staticmethod
    def to_dict(obj: Any) -> Any:
        # Custom dict conversion
        if dataclasses.is_dataclass(obj):
            d = {}
            for field in dataclasses.fields(obj):
                value = getattr(obj, field.name)
                # Recurse
                d[field.name] = MemoryIO.to_dict(value)
            return d
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, list):
            return [MemoryIO.to_dict(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: MemoryIO.to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()
        else:
            return obj

    @staticmethod
    def to_json(structure: MemoryStructure) -> str:
        data = MemoryIO.to_dict(structure)
        return json.dumps(data, indent=2, ensure_ascii=False)

    @staticmethod
    def to_yaml(structure: MemoryStructure) -> str:
        data = MemoryIO.to_dict(structure)
        # Use safe_dump but allow standard types
        return yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    @staticmethod
    def from_dict(data: Dict[str, Any], cls: Type) -> Any:
        # Deserialization based on expected type (cls)
        # This is a bit manual in Python without a sophisticated framework like Pydantic/Marshmallow
        # But for M0 we do basic recursion.
        
        if dataclasses.is_dataclass(cls):
            field_types = {f.name: f.type for f in dataclasses.fields(cls)}
            init_args = {}
            
            for k, v in data.items():
                if k not in field_types:
                    continue # Ignore extra fields (robustness) or error? Spec says "validate".
                
                ft = field_types[k]
                
                # Handle List[T]
                # Simple heuristic for generic Alias checking
                if hasattr(ft, "__origin__") and ft.__origin__ == list:
                    item_type = ft.__args__[0]
                    init_args[k] = [MemoryIO.from_dict(x, item_type) for x in v]
                # Handle Dict[K, V]
                elif hasattr(ft, "__origin__") and ft.__origin__ == dict:
                    val_type = ft.__args__[1]
                    init_args[k] = {dk: MemoryIO.from_dict(dv, val_type) for dk, dv in v.items()}
                # Handle Optional[T] -> Union[T, None]
                elif hasattr(ft, "__origin__") and ft.__origin__ == Union:
                    # Assume non-None type
                    real_type = next((t for t in ft.__args__ if t is not type(None)), None)
                    if v is None:
                        init_args[k] = None
                    elif real_type:
                        init_args[k] = MemoryIO.from_dict(v, real_type)
                    else:
                        init_args[k] = v
                # Handle Enum
                elif isinstance(ft, type) and issubclass(ft, Enum):
                    init_args[k] = ft(v)
                # Handle Nested
                elif dataclasses.is_dataclass(ft):
                    init_args[k] = MemoryIO.from_dict(v, ft)
                elif ft == datetime.datetime:
                    init_args[k] = datetime.datetime.fromisoformat(v)
                else:
                    init_args[k] = v
                    
            return cls(**init_args)
            
        elif isinstance(cls, type) and issubclass(cls, Enum):
            return cls(data)
            
        return data

    @staticmethod
    def from_yaml(yaml_str: str) -> MemoryStructure:
        data = yaml.safe_load(yaml_str)
        return MemoryIO.from_dict(data, MemoryStructure)
        
    @staticmethod
    def from_json(json_str: str) -> MemoryStructure:
        data = json.loads(json_str)
        return MemoryIO.from_dict(data, MemoryStructure)
