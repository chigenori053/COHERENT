"""
Logic Controller (Core B Facade)
Responsible for:
- Orchestration of Logic Engines
- Dynamic Task Routing
- Registry of Capabilities
"""

import logging
from typing import Any, Dict, Optional, Callable, Type
from dataclasses import dataclass

from .computation.arithmetic_engine import ArithmeticEngine
from .computation.symbolic_engine import SymbolicEngine
from .validation.validation_engine import ValidationEngine

from coherent.core.computation_engine import ComputationEngine

@dataclass
class LogicTask:
    task_type: str  # e.g., "compute", "verify", "validate"
    content: Any
    context: Optional[Dict] = None
    engine_hint: Optional[str] = None

class LogicController:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._engines: Dict[str, Any] = {}
        
        # Instantiate Engines
        self.symbolic = SymbolicEngine()
        self.arithmetic = ArithmeticEngine()
        self.computation = ComputationEngine(self.symbolic)
        self.validator = ValidationEngine(self.computation)
        
        # Register default engines
        self.register_engine("arithmetic", self.arithmetic)
        self.register_engine("symbolic", self.symbolic)
        self.register_engine("computation", self.computation)
        self.register_engine("validation", self.validator)
        
        # Default Routing Table
        self._routing = {
            "compute": "symbolic", # Default to symbolic for general compute
            "verify": "validation",
            "arithmetic": "arithmetic"
        }

    def register_engine(self, name: str, engine_instance: Any):
        """Register a new logic engine capability."""
        self._engines[name] = engine_instance
        self.logger.info(f"Registered logic engine: {name}")

    def orchestrate(self, task: LogicTask) -> Any:
        """
        Dynamic Orchestration of Logic Tasks.
        """
        try:
            # 1. Select Engine
            engine_name = task.engine_hint or self._routing.get(task.task_type)
            if not engine_name or engine_name not in self._engines:
                raise ValueError(f"No suitable engine found for task: {task.task_type} (hint: {task.engine_hint})")
                
            engine = self._engines[engine_name]
            
            # 2. Dispatch
            self.logger.info(f"Dispatching task {task.task_type} to {engine_name}")
            
            if task.task_type == "compute":
                # Assuming engine has evaluate or similar
                if hasattr(engine, 'evaluate'):
                    return engine.evaluate(task.content, task.context or {})
                elif hasattr(engine, 'numeric_eval'):
                     return engine.numeric_eval(task.content, task.context or {})
                     
            elif task.task_type == "verify":
                 # context is key for verification
                 return engine.validate_step(task.content.get('before'), task.content.get('after'), context=task.context)
                 
            elif task.task_type == "arithmetic":
                 # Direct atomic Ops
                 pass
                 
            # Fallback / Generic Call
            return f"Executed {task.task_type} on {engine_name}"

        except Exception as e:
            self.logger.error(f"Orchestration failed: {e}")
            raise e

    # --- Legacy Facade Methods (for backward compatibility if needed, or wrappers) ---
    def verify(self, hypothesis: Any, context: Dict) -> bool:
        """
        Verify wrapper.
        """
        # Example: Convert to Task
        task = LogicTask(task_type="verify", content=hypothesis, context=context)
        return self.orchestrate(task)

    def execute_compute(self, expression: str) -> str:
        """
        Compute wrapper.
        """
        task = LogicTask(task_type="compute", content=expression)
        return self.orchestrate(task)
