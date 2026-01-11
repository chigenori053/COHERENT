from enum import Enum
from typing import List, Optional, Any, Dict, Union
from pydantic import BaseModel, Field, ConfigDict

class IntentType(str, Enum):
    SOLVE = "solve"
    VERIFY = "verify"
    HINT = "hint"
    EXPLAIN = "explain"

class MathDomain(str, Enum):
    ARITHMETIC = "arithmetic"
    ALGEBRA = "algebra"
    CALCULUS = "calculus"
    LINEAR_ALGEBRA = "linear_algebra"
    GEOMETRY = "geometry"
    STATISTICS = "statistics"
    UNKNOWN = "unknown"

class GoalType(str, Enum):
    FINAL_VALUE = "final_value"
    TRANSFORMATION = "transformation"
    PROOF = "proof"

class Goal(BaseModel):
    type: GoalType
    target: Optional[str] = None # e.g., variable to solve for, or form to transform to

class InputItemType(str, Enum):
    EXPRESSION = "expression"
    EQUATION = "equation"
    TEXT = "text"
    Data = "data"

class InputItem(BaseModel):
    type: InputItemType
    value: Any
    metadata: Optional[Dict[str, Any]] = None

class Constraints(BaseModel):
    symbolic_only: bool = False
    steps_required: bool = True
    allow_approximation: bool = False
    max_steps: Optional[int] = None

class LanguageMeta(BaseModel):
    original_language: str = "en"
    ambiguity_score: float = 0.0
    detected_intent_confidence: float = 1.0

class SemanticIR(BaseModel):
    """
    Semantic Intermediate Representation (SIR)
    Defines the structured semantics of a natural language request.
    """
    task: IntentType
    math_domain: MathDomain = MathDomain.UNKNOWN
    goal: Optional[Goal] = None
    inputs: List[InputItem] = Field(default_factory=list)
    constraints: Constraints = Field(default_factory=Constraints)
    explanation_level: int = 0  # 0: minimal, 1: standard, 2: detailed
    language_meta: LanguageMeta = Field(default_factory=LanguageMeta)
    
    model_config = ConfigDict(use_enum_values=True)
