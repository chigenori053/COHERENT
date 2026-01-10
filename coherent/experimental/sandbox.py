import os
import json
import time
from typing import Any, Dict, Optional, Union
from pathlib import Path
from dataclasses import asdict

class Sandbox:
    """
    Sandbox Environment for non-invasive evaluation of Core Architecture.
    Isolates experimental outputs and logs from Core decision-making.
    """
    def __init__(self, root_dir: str = None):
        """
        Initialize Sandbox.
        
        Args:
            root_dir: Root directory for sandbox artifacts. 
                      Defaults to 'experimental/sandbox' in the project root.
        """
        if root_dir:
            self.root_dir = Path(root_dir)
        else:
            # Default to <project_root>/experimental/sandbox
            # Assuming this code is in <project_root>/coherent/experimental/sandbox.py
            # So generic fallback or specific path
            base = Path(os.getcwd())
            self.root_dir = base / "experimental" / "sandbox"
            
        self.outputs_dir = self.root_dir / "outputs"
        self.logs_dir = self.root_dir / "logs"
        self.traces_dir = self.root_dir / "traces"
        self.metadata_dir = self.root_dir / "metadata"
        
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure all sandbox directories exist."""
        for d in [self.outputs_dir, self.logs_dir, self.traces_dir, self.metadata_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def capture_input(self, state: Any, metadata: Dict[str, Any] = None):
        """
        Capture input state details.
        
        Args:
            state: Input state (e.g. numpy array)
            metadata: Input metadata
        """
        try:
            data = {
                "timestamp": time.time(),
                "type": "input",
                "metadata": metadata if metadata else {}
            }
            # Note: State capturing might be heavy if large arrays. 
            # For v1, we log metadata and maybe shape.
            if hasattr(state, "shape"):
                data["state_shape"] = str(state.shape)
            
            self._write_log("inputs", data)
        except Exception as e:
            # Sandbox MUST NOT fail Core execution
            print(f"[Sandbox] Error capturing input: {e}")

    def capture_decision(self, decision_state: Any, action: Any, metadata: Dict[str, Any] = None):
        """
        Capture decision logic and outcome.
        
        Args:
            decision_state: Internal state used for decision (DecisionState objects)
            action: Resulting action (Action enum)
            metadata: Additional context
        """
        try:
            data = {
                "timestamp": time.time(),
                "type": "decision",
                "action": str(action),
                "metadata": metadata if metadata else {}
            }
            
            # Serialize decision_state if it's a dataclass
            if hasattr(decision_state, "__dataclass_fields__"):
                data["decision_state"] = asdict(decision_state)
            else:
                data["decision_state"] = str(decision_state)
                
            self._write_log("decisions", data)
        except Exception as e:
            print(f"[Sandbox] Error capturing decision: {e}")

    def _write_log(self, category: str, data: Dict[str, Any]):
        """
        Append structured log to file.
        
        Args:
            category: Log category (filename prefix)
            data: Dictionary to serialize
        """
        filename = f"{category}.jsonl"
        filepath = self.logs_dir / filename
        
        def default_serializer(obj):
            if hasattr(obj, "tolist"):
                return obj.tolist()
            return str(obj)

        try:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, default=default_serializer) + "\n")
        except Exception as e:
            print(f"[Sandbox] Write error: {e}")
