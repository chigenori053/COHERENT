from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
from pydantic import BaseModel, Field

from coherent.tools.language.models import SemanticIR
from coherent.core.memory.optical_store import OpticalFrequencyStore
from coherent.core.multimodal.text_encoder import HolographicTextEncoder

class ExperienceUnit(BaseModel):
    """
    Represents a unit of experience: a solved problem or processed instruction.
    Stored in Optical Memory for recall.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query_text: str
    sir: SemanticIR
    result: Dict[str, Any]
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    confidence: float = 1.0

class ExperienceManager:
    """
    Manages the storage and recall of ExperienceUnits using Optical Holographic Memory.
    """
    
    def __init__(self, optical_store: Optional[OpticalFrequencyStore] = None):
        # If no store provided, we might initialize a new one or stay disabled
        self.optical_store = optical_store
        self.encoder = HolographicTextEncoder() if optical_store else None
        
    def is_enabled(self) -> bool:
        if self.optical_store is None or self.encoder is None:
            return False
            
        # Check if SentenceTransformer is actually available
        from coherent.core.multimodal.text_encoder import SentenceTransformer
        return SentenceTransformer is not None

    def store_experience(self, query: str, sir: SemanticIR, result: Dict[str, Any]) -> str:
        """
        Stores a new experience unit.
        """
        if not self.is_enabled():
            return ""

        experience = ExperienceUnit(
            query_text=query,
            sir=sir,
            result=result
        )
        
        # Vectorize the query text using Holographic Encoder
        # text_encoder.encode returns a complex HolographicTensor (spectrum)
        # OpticalFrequencyStore expects List[float] currently for 'vectors' argument if generic, 
        # BUT looking at optical_store.py:
        # _encode_signal takes List[List[float]] and does FFT itself. 
        # IT seems OpticalFrequencyStore takes REAL vectors and converts them.
        # So we should use a real-valued embedding here (e.g. from TransformerEncoder style or intermediate).
        # HolographicTextEncoder in text_encoder.py has .encode returning HolographicTensor (complex).
        # Let's check text_encoder.py again.
        # It has a TransformerEncoder legacy class too.
        # If OpticalFrequencyStore._encode_signal does the FFT, we should feed it real vectors.
        # So we need to access the real embedding, NOT the complex spectrum from HolographicTextEncoder.
        
        # Let's use the 'TransformerEncoder' logic or extracting embedding from HolographicTextEncoder if possible.
        # Looking at text_encoder.py, HolographicTextEncoder.encode calls model.encode().
        # We might need to subclass or expose the real embedding.
        # For now, let's instantiate a TransformerEncoder separately or rely on its logic.
        
        # Actually, if we use HolographicTextEncoder, we get complex tensor. 
        # OpticalFrequencyStore.add expects "vectors: List[List[float]]".
        # It then calls self._encode_signal(vectors).
        # So yes, we need real-valued vectors.
        
        # We will use the TransformerEncoder (legacy) to get the real vector.
        # Or better, just copy the logic since we want to be clean.
        # Let's use the existing TransformerEncoder in text_encoder.py
        from coherent.core.multimodal.text_encoder import TransformerEncoder
        self.real_encoder = TransformerEncoder()
        
        vector = self.real_encoder.encode(query)
        if not vector:
             return ""
             
        # Normalize metadata
        # We store the ExperienceUnit as dict in metadata
        meta = experience.model_dump()
        
        self.optical_store.add(
            collection_name="experiences",
            vectors=[vector],
            metadatas=[meta],
            ids=[experience.id]
        )
        
        return experience.id

    def recall_experience(self, query: str, threshold: float = 0.95) -> Optional[ExperienceUnit]:
        """
        Recall a similar experience if it matches strongly.
        Returns the ExperienceUnit with 'ambiguity' injected into its result or metadata if possible,
        or we might need to change return type. For minimal breaking change, we attach it to the unit instance properties if allowed,
        or return a tuple.
        
        Let's modify ExperienceUnit or just set an attribute on the returned object dynamically 
        since Pydantic models are not friendly to dynamic attrs by default unless Config allows it.
        
        Actually, let's just return the ambiguity score as part of the result if we can.
        Or better, let's wrap it?
        
        For now, we'll attach it to the `result` dict of the experience unit before returning.
        """
        if not self.is_enabled():
            return None
            
        from coherent.core.multimodal.text_encoder import TransformerEncoder
        if not hasattr(self, 'real_encoder'):
             self.real_encoder = TransformerEncoder()
             
        vector = self.real_encoder.encode(query)
        if not vector:
            return None
            
        # Query store
        results = self.optical_store.query(
            collection_name="experiences",
            query_vec=vector,
            top_k=5 # We need top k to check for ambiguity if the store calculates it based on distribution
        )
        
        if not results:
            return None
            
        best = results[0]
        score = best['score']
        ambiguity = best.get('ambiguity', 0.0)
        
        # Check resonance/distance/score. 
        if score >= threshold:
            try:
                data = best['metadata']
                unit = ExperienceUnit(**data)
                # Inject ambiguity into the result so CoreRuntime can see it
                # We do this to avoid changing the method signature too much for V1
                if isinstance(unit.result, dict):
                    unit.result['ambiguity_score'] = ambiguity
                    unit.result['match_score'] = score
                return unit
            except Exception:
                return None
                
        return None
