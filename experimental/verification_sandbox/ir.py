"""
IR Definition for Verification Sandbox
Defines the minimal abstract nodes for structural verification.
"""

from typing import List, Optional, Any
from dataclasses import dataclass

@dataclass
class IRNode:
    pass

@dataclass
class Value(IRNode):
    content: str  # e.g., "10", "x"

@dataclass
class Assignment(IRNode):
    target: str
    value: Any # Value or expression string

@dataclass
class Exit(IRNode):
    pass

@dataclass
class Sequence(IRNode):
    nodes: List[IRNode]

@dataclass
class If(IRNode):
    condition: str
    body: Sequence
    else_body: Optional[Sequence] = None

@dataclass
class Loop(IRNode):
    condition: str
    body: Sequence
