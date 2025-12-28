from __future__ import annotations
from typing import List, Dict, Any, Optional, Literal
from uuid import UUID
from pydantic import BaseModel, Field

# --- Base Event ---
class TraceEvent(BaseModel):
    trace_id: str
    step_id: int
    event_seq: int
    event_type: str
    timestamp_ms: int

# --- Payload Models ---

class StepInputEvent(TraceEvent):
    event_type: Literal["STEP_INPUT"] = "STEP_INPUT"
    input_text: str
    normalized_text: str

class AstGeneralizedEvent(TraceEvent):
    event_type: Literal["AST_GENERALIZED"] = "AST_GENERALIZED"
    ast_generalized: str
    ast_features: Dict[str, Any]

class ComplexVector(BaseModel):
    real: List[float]
    imag: List[float]

class EncodeEndEvent(TraceEvent):
    event_type: Literal["ENCODE_END"] = "ENCODE_END"
    H_complex: ComplexVector
    encoding_meta: Dict[str, Any]

class RecallItem(BaseModel):
    mem_id: str
    resonance: float

class RecallTopKEvent(TraceEvent):
    event_type: Literal["RECALL_TOPK"] = "RECALL_TOPK"
    query_id: str
    topK: List[RecallItem]
    theta: float

class InterferenceItem(BaseModel):
    mem_id: str
    complex: ComplexVector

class InterferenceTopKEvent(TraceEvent):
    event_type: Literal["INTERFERENCE_TOPK"] = "INTERFERENCE_TOPK"
    interference: List[InterferenceItem]

class ValidationResultEvent(TraceEvent):
    event_type: Literal["VALIDATION_RESULT"] = "VALIDATION_RESULT"
    decision: Literal["ACCEPT", "REVIEW", "REJECT"]
    confidence: float
    reason_codes: List[str]

class MemWriteEvent(TraceEvent):
    event_type: Literal["MEM_WRITE"] = "MEM_WRITE"
    mem_id: str
    H_complex: ComplexVector
    tags: Dict[str, Any]
