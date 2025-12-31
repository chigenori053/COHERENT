from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging

class MemoryStore(ABC):
    """
    Abstract base class for all memory sub-areas (Accept, Review, Reject).
    Enforces the interface for writing and recalling memories.
    """
    def __init__(self, store_id: str):
        self.store_id = store_id
        self.data: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"coherent.memory.{store_id}")

    @abstractmethod
    def write(self, key: str, value: Any) -> None:
        """Writes data to the store."""
        pass

    @abstractmethod
    def recall(self, query: Any, context: Optional[str] = None) -> Any:
        """
        Recalls data from the store.
        'context' is used for safety guards (e.g. reviewing restricted stores).
        """
        pass

    def count(self) -> int:
        return len(self.data)


class AcceptStore(MemoryStore):
    """
    Stores verified knowledge.
    Role: Learning & Recall.
    Resonance Participation: Full.
    """
    def __init__(self):
        super().__init__("AcceptStore")

    def write(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.logger.info(f"Written to AcceptStore: {key}")

    def recall(self, query: Any, context: Optional[str] = None) -> Any:
        # Full recall enabled.
        # In a real implementation, this would perform resonance calculation.
        return self.data


class ReviewStore(MemoryStore):
    """
    Stores uncertain or boundary cases.
    Role: Evaluation only.
    Recall: Conditional (Requires explicit 'evaluation' context).
    """
    ALLOWED_CONTEXTS = {"threshold_tuning", "utility_analysis", "human_override_review"}

    def __init__(self):
        super().__init__("ReviewStore")

    def write(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.logger.info(f"Flagged for Review: {key}")

    def recall(self, query: Any, context: Optional[str] = None) -> Any:
        # Guard-4: Review Recall Restriction
        if context not in self.ALLOWED_CONTEXTS:
            self.logger.warning(f"Access denied to ReviewStore with context: {context}")
            return {} # Or raise PermissionError
        
        return self.data


class RejectStore(MemoryStore):
    """
    Stores invalid or harmful patterns.
    Role: Counter-examples.
    Recall: Forbidden for standard operations.
    """
    def __init__(self):
        super().__init__("RejectStore")

    def write(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.logger.info(f"Rejected: {key}")

    def recall(self, query: Any, context: Optional[str] = None) -> Any:
        # Guard-5: Reject Store Full Isolation
        # Guard-1: Resonance Isolation (No resonance mixing)
        if context == "counter_example_logging":
            return self.data
        
        self.logger.warning("Attempted to recall from RejectStore - Access Denied.")
        return {}
