import time
import uuid
import random
import numpy as np
from typing import Generator
from .models import (
    TraceEvent, StepInputEvent, AstGeneralizedEvent, EncodeEndEvent, 
    RecallTopKEvent, InterferenceTopKEvent, ValidationResultEvent,
    ComplexVector, RecallItem, InterferenceItem
)

class TraceSimulator:
    def __init__(self):
        self.trace_id = str(uuid.uuid4())
        self.step_id = 1
        self.event_seq = 0
        self.rng = np.random.default_rng()

    def _next_header(self, event_type: str) -> dict:
        self.event_seq += 1
        return {
            "trace_id": self.trace_id,
            "step_id": self.step_id,
            "event_seq": self.event_seq,
            "event_type": event_type,
            "timestamp_ms": int(time.time() * 1000)
        }

    def _gen_complex_vector(self, size=1024, pattern="random"):
        real = self.rng.standard_normal(size)
        imag = self.rng.standard_normal(size)
        if pattern == "pulse":
            # Add some structure
            idx = self.rng.integers(0, size, 10)
            real[idx] += 5.0
        return ComplexVector(real=real.tolist(), imag=imag.tolist())

    def generate_step_sequence(self) -> Generator[TraceEvent, None, None]:
        # 1. STEP_INPUT
        yield StepInputEvent(
            **self._next_header("STEP_INPUT"),
            input_text="3x + 5 = 11",
            normalized_text="3*x + 5 = 11"
        )
        time.sleep(0.1)

        # 2. AST_GENERALIZED
        yield AstGeneralizedEvent(
            **self._next_header("AST_GENERALIZED"),
            ast_generalized="a*v + b = c",
            ast_features={"node_count": 7, "depth": 3}
        )
        time.sleep(0.1)

        # 3. ENCODE_END
        yield EncodeEndEvent(
            **self._next_header("ENCODE_END"),
            H_complex=self._gen_complex_vector(pattern="pulse"),
            encoding_meta={"method": "FFT", "target_dim": 1024}
        )
        time.sleep(0.2)

        # 4. RECALL_TOPK
        topk = [
             RecallItem(mem_id=f"m-{self.rng.integers(100, 999)}", resonance=0.9),
             RecallItem(mem_id=f"m-{self.rng.integers(100, 999)}", resonance=0.6)
        ]
        yield RecallTopKEvent(
            **self._next_header("RECALL_TOPK"),
            query_id=str(uuid.uuid4()),
            topK=topk,
            theta=0.7
        )
        time.sleep(0.2)

        # 5. INTERFERENCE_TOPK
        int_items = [
            InterferenceItem(
                mem_id=item.mem_id,
                complex=self._gen_complex_vector()
            ) for item in topk 
        ]
        yield InterferenceTopKEvent(
            **self._next_header("INTERFERENCE_TOPK"),
            interference=int_items
        )
        time.sleep(0.2)

        # 6. VALIDATION_RESULT
        yield ValidationResultEvent(
            **self._next_header("VALIDATION_RESULT"),
            decision="ACCEPT",
            confidence=0.87,
            reason_codes=["RECALL_MATCH"]
        )
        
        self.step_id += 1
