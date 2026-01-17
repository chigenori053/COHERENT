
import logging
from typing import List, Optional, Dict, Any
import uuid
from .vector_store import VectorStoreBase
from .schema import ExperienceEntry

class ExperienceManager:
    """
    Manages the "Experience Network" (Edges between AST States).
    Handles saving and retrieving ExperienceEntry items from the VectorStore.
    """
    def __init__(self, vector_store: VectorStoreBase):
        self.vector_store = vector_store
        self.collection_name = "experience_network"
        self.logger = logging.getLogger(__name__)

    def save_edge(self, source_state_gen: str, target_state_gen: str, rule_id: str, source_vector: List[float]):
        """
        Saves a transition edge (Action) from Source to Target.
        
        Args:
            source_state_gen: Generalized string of source state.
            target_state_gen: Generalized string of target state.
            rule_id: The action taken.
            source_vector: Embedding of the source state.
        """
        edge_id = str(uuid.uuid4())
        
        entry = ExperienceEntry(
            id=edge_id,
            original_expr=source_state_gen, # Storing generalized form as the 'original' for lookup
            next_expr=target_state_gen,
            rule_id=rule_id,
            result_label="EXACT", # Default for now
            category="math",      # Default
            score=1.0,
            vector=source_vector
        )
        
        self.vector_store.add(
            collection_name=self.collection_name,
            vectors=[source_vector],
            metadatas=[entry.to_metadata()],
            ids=[edge_id]
        )
        self.logger.info(f"Saved Edge: {source_state_gen} -> [{rule_id}] -> {target_state_gen}")

    def find_similar_edges(self, query_vector: List[float], top_k: int = 5) -> List[ExperienceEntry]:
        """
        Recall: Finds edges starting from states similar to the query.
        """
        results = self.vector_store.query(
            collection_name=self.collection_name,
            query_vec=query_vector,
            top_k=top_k
        )
        
        entries = []
        for res in results:
            meta = res.get("metadata", {})
            try:
                entry = ExperienceEntry(
                    id=res["id"],
                    original_expr=meta.get("original_expr", ""),
                    next_expr=meta.get("next_expr", ""),
                    rule_id=meta.get("rule_id", ""),
                    result_label=meta.get("result_label", "EXACT"),
                    category=meta.get("category", "math"),
                    score=meta.get("score", 0.0),
                    vector=None, # Not returned by query usuall
                    metadata=meta
                )
                entries.append(entry)
            except Exception as e:
                self.logger.error(f"Failed to parse experience entry: {e}")
                
        return entries

    def save_refusal(self, decision_state: Any, action: str, metadata: Dict[str, Any]):
        """
        Persists a refusal or review decision (SUPPRESS/DEFER_REVIEW) as an experience.
        This allows the system to recall 'what it refused' and 'why'.
        """
        edge_id = str(uuid.uuid4())
        
        # Original input content or ID
        original_expr = metadata.get("content", metadata.get("id", "unknown_input"))
        
        entry = ExperienceEntry(
            id=edge_id,
            original_expr=original_expr,
            next_expr="<BLOCKED>",
            rule_id=action,     # e.g., "Action.SUPPRESS"
            result_label="REJECTED" if "SUPPRESS" in action else "REVIEW_NEEDED",
            category="decision_trace",
            score=decision_state.resonance_score if hasattr(decision_state, 'resonance_score') else 0.0,
            vector=None, # storing validation vector if available? For now None or need to pass input vector.
            metadata={
                "decision_margin": decision_state.margin if hasattr(decision_state, 'margin') else 0.0,
                "entropy": decision_state.entropy_estimate if hasattr(decision_state, 'entropy_estimate') else 0.0,
                "raw_metadata": metadata
            }
        )
        
        # note: saving without vector for now means it's not recallable by similarity 
        # unless we pass the input vector. 
        # Ideally we should pass 'input_vector' to save_refusal.
        # But for V1.1 verification of storage, this suffices.
        
        self.vector_store.add(
            collection_name=self.collection_name,
            vectors=[[0.0]*64], # Placeholder vector or need real one.
            metadatas=[entry.to_metadata()],
            ids=[edge_id]
        )
        self.logger.info(f"Saved Refusal: {action} for {original_expr}")

    def log_experience(self, input_signal: Any, metadata: Dict[str, Any]):
        """
        Generic logging of cognitive experience (Decisions).
        """
        edge_id = str(uuid.uuid4())
        
        # Extract content
        content = ""
        if isinstance(input_signal, dict):
            content = input_signal.get("text", str(input_signal))
            # If image involved, maybe append?
            if "image_name" in input_signal:
                content += f" [Image: {input_signal['image_name']}]"
        else:
            content = str(input_signal)
            
        decision_val = metadata.get("decision", "UNKNOWN")
        
        entry = ExperienceEntry(
            id=edge_id,
            original_expr=content,
            next_expr="<DECISION>",
            rule_id=f"Decision.{decision_val}",
            result_label=decision_val,
            category="cognitive_trace",
            score=metadata.get("metrics", {}).get("C", 0.0), # Confidence as score
            vector=None, # Ideal: input vector
            metadata=metadata
        )
        
        self.vector_store.add(
            collection_name=self.collection_name,
            vectors=[[0.0]*64], # Placeholder vector
            metadatas=[entry.to_metadata()],
            ids=[edge_id]
        )
        self.logger.info(f"Logged Experience: {decision_val} for {content}")
