from dataclasses import dataclass, field
from typing import Any, Dict
import time

@dataclass
class VariantLink:
    primary_id: str
    variant_id: str
    relationship_type: str = "soft_variant"
    similarity_score: float = 0.0
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, primary: Any, variant: Any, metadata: Dict[str, Any] = None) -> 'VariantLink':
        """
        Creates a link structure. 
        Note: The actual 'hologram' objects might need to expose an ID.
        Here we assume they do, or we generate one.
        """
        p_id = getattr(primary, 'id', str(id(primary)))
        v_id = getattr(variant, 'id', str(id(variant)))
        
        return cls(
            primary_id=p_id, 
            variant_id=v_id, 
            metadata=metadata or {}
        )
