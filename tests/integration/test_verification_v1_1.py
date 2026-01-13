
import unittest
import numpy as np
import os
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import asdict

from coherent.core.memory.holographic.dynamic import DynamicHolographicMemory
from coherent.core.memory.holographic.static import StaticHolographicMemory, StaticEvaluationProfile
from coherent.core.memory.holographic.causal import CausalHolographicMemory, DecisionState, Action
from coherent.core.memory.holographic.orchestrator import MemoryOrchestrator
from coherent.experimental.sandbox import Sandbox

# --- Mock/Test Components ---

class TestSandbox(Sandbox):
    """
    Captures interactions for verification without external dependencies.
    """
    def __init__(self):
        self.captured_inputs = []
        self.captured_decisions = []
        self.logs = []

    def capture_input(self, state: np.ndarray, metadata: Dict[str, Any] = None):
        self.captured_inputs.append({"state": state, "metadata": metadata})

    def capture_decision(self, decision_state, action, metadata):
        self.captured_decisions.append({
            "decision_state": decision_state,
            "action": action,
            "metadata": metadata
        })
    
    def log(self, entry: Dict[str, Any]):
        self.logs.append(entry)

class MockSemanticParser:
    """
    Simple mock to map test strings to consistent vectors.
    Dimensions:
    0: Math/Calculation
    1: 3x (Symbol x)
    2: 5x
    3: Poetry/Creative
    4: Undefined/Error
    5: Sequence/Order
    6: Ambiguity
    """
    def parse_to_vector(self, text: str) -> np.ndarray:
        vec = np.zeros(64)
        normalized = text.lower()
        
        # Base "Math" context for valid instructions
        if any(x in normalized for x in ["計算", "calculate", "まと", "solve", "math", "x", "式", "引", "割"]):
            vec[0] = 0.8 # Strong Math component
            
        if "3x" in normalized: vec[1] = 1.0
        if "5x" in normalized: vec[2] = 1.0
        if "7x" in normalized: 
            # 7x shares "x" property, so let's map it to similar as 3x/5x for test sake or just rely on 'vec[0]'
            vec[1] = 0.5; vec[2] = 0.5 
            
        if "poetry" in normalized or "詩的" in normalized: 
            vec[0] = 0.0 # Remove math context
            vec[3] = 1.0
            
        if "undefined" in normalized or "未定義" in normalized:
            vec[4] = 1.0
            
        if "sequence" in normalized or "順序" in normalized or "まず" in normalized:
            vec[5] = 1.0
            
        if "いい感じ" in normalized or "nicely" in normalized:
            vec[6] = 1.0 # Ambiguity char
            vec[0] = 0.2 # Weak math
            
        # Add slight noise for realism, but keep it small to preserve resonance
        np.random.seed(hash(text) % 2**32)
        vec += np.random.normal(0, 0.01, 64)
        
        return vec / (np.linalg.norm(vec) + 1e-9)

class MockExperienceManager:
    def __init__(self):
        self.saved_refusals = []

    def save_refusal(self, decision_state, action, metadata):
        self.saved_refusals.append({
            "state": decision_state,
            "action": action,
            "metadata": metadata
        })

class TestVerificationV1_1(unittest.TestCase):
    
    def setUp(self):
        # 3. System Configuration
        self.dynamic = DynamicHolographicMemory(capacity=50)
        self.static = StaticHolographicMemory()
        
        # Inject Mock Experience Manager for Persistence Test
        self.experience_manager = MockExperienceManager()
        self.causal = CausalHolographicMemory(experience_manager=self.experience_manager)
        
        self.sandbox = TestSandbox()
        
        self.orchestrator = MemoryOrchestrator(
            dynamic=self.dynamic,
            static=self.static,
            causal=self.causal,
            sandbox=self.sandbox
        )
        
        self.parser = MockSemanticParser()
        self.report_data = []
        
        self._bootstrap_system()

    def _bootstrap_system(self):
        """Seed the system with basic Math concept to allow resonance."""
        # Concept: Algebra/Math
        # Vector: High in dimension 0
        vec = np.zeros(64)
        vec[0] = 1.0
        vec = vec / np.linalg.norm(vec)
        
        # Manually adding to Static Memory
        self.static.add(vec, {"id": "concept_math", "range": "universal"})
        print("[Setup] Bootstrapped Static Memory with 'concept_math'")

    def _log_result(self, test_id, input_text, decision, result, details=None):
        entry = {
            "test_id": test_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "input_text": input_text,
            "decision": str(decision),
            "result": result,
            "details": details or {}
        }
        self.report_data.append(entry)
        print(f"[{test_id}] {result}: {decision} | {input_text} | {details}")

    def test_execution(self):
        print("\n=== Starting Verification & Learning Test v1.1 ===\n")
        
        # --- T1 Direct Instruction ---
        self._run_t1()
        
        # --- T2 Paraphrase ---
        self._run_t2()
        
        # --- T3 Causal ---
        self._run_t3()
        
        # --- T4 Ambiguity ---
        self._run_t4()
        
        # --- T5 Invalid ---
        self._run_t5()
        
        # --- T6 Recall Failure ---
        self._run_t6()
        
        # --- T7 Experience Reuse ---
        self._run_t7()
        
        # Generate Report
        self._generate_report()

    def _run_t1(self):
        """T1 — Direct Instruction Recognition"""
        input_text = "3x + 5x を計算せよ"
        vec = self.parser.parse_to_vector(input_text)
        meta = {"id": "T1_REQ", "content": input_text, "validation_confidence": 1.0}
        
        # Execute
        self.orchestrator.process_input(vec, meta)
        
        # Verify
        last_decision = self.sandbox.captured_decisions[-1]
        action = last_decision["action"]
        res = last_decision["decision_state"].resonance_score
        
        # With bootstrap, it should resonate with "Math" concept.
        # Expectation: RETAIN or PROMOTE (if resonance is high enough)
        # Logit will be better.
        
        result = "PASS" if action in [Action.RETAIN, Action.PROMOTE] else "FAIL"
        
        self._log_result("T1", input_text, action, result, {
            "prob": last_decision["decision_state"].resonance_score
        })
        
        # Force promote specific skill for T2/T7
        self.orchestrator.promote_to_static(vec, {"id": "skill_simplify_algebra", "content": "3x+5x"})

    def _run_t2(self):
        """T2 — Paraphrase Robustness"""
        # "3x における 5x ..." - similar vector
        input_text = "3x と 5x をまとめて"
        vec = self.parser.parse_to_vector(input_text)
        meta = {"id": "T2_REQ", "content": input_text}
        
        self.orchestrator.process_input(vec, meta)
        
        # Should resonate with T1 promoted item
        last_decision = self.sandbox.captured_decisions[-1]
        resonance = last_decision["decision_state"].resonance_score
        
        # Expectation: High resonance
        result = "PASS" if resonance > 0.6 else "FAIL" 
        self._log_result("T2", input_text, last_decision["action"], result, {"resonance": resonance})

    def _run_t3(self):
        """T3 — Multi-step Causal Instruction"""
        input_text = "まず両辺から3を引き、その後2で割れ"
        vec = self.parser.parse_to_vector(input_text)
        meta = {"id": "T3_REQ", "content": input_text, "validation_confidence": 0.95}
        
        self.orchestrator.process_input(vec, meta)
        
        last_decision = self.sandbox.captured_decisions[-1]
        # Should resonate with Math
        result = "PASS" if last_decision["action"] != Action.SUPPRESS else "FAIL"
        self._log_result("T3", input_text, last_decision["action"], result)
        
        self.causal.add_transition(vec, vec) 

    def _run_t4(self):
        """T4 — Ambiguous Instruction"""
        input_text = "いい感じに解いて" # "Solve nicely"
        vec = self.parser.parse_to_vector(input_text)
        
        self.orchestrator.process_input(vec, {"id": "T4_REQ", "content": input_text})
        last_decision = self.sandbox.captured_decisions[-1]
        
        result = "PASS" if last_decision["action"] in [Action.DEFER_REVIEW, Action.SUPPRESS] else "FAIL" 
        
        # Verify Persistence
        has_persisted = any(r["metadata"]["content"] == input_text for r in self.experience_manager.saved_refusals)
        if not has_persisted:
            result = "FAIL_PERSISTENCE"
            
        self._log_result("T4", input_text, last_decision["action"], result, {"persisted": has_persisted})

    def _run_t5(self):
        """T5 — Invalid Instruction"""
        input_text = "この式を詩的に説明して" 
        vec = self.parser.parse_to_vector(input_text) 
        
        self.orchestrator.process_input(vec, {"id": "T5_REQ", "content": input_text})
        last_decision = self.sandbox.captured_decisions[-1]
         
        result = "PASS" if last_decision["action"] == Action.SUPPRESS else "FAIL"
        
        # Verify Persistence
        has_persisted = any(r["metadata"]["content"] == input_text for r in self.experience_manager.saved_refusals)
        if not has_persisted:
            result = "FAIL_PERSISTENCE"
        
        self._log_result("T5", input_text, last_decision["action"], result, {"persisted": has_persisted})

    def _run_t6(self):
        """T6 — Recall Failure"""
        input_text = "未定義の演算Zを実行せよ"
        vec = self.parser.parse_to_vector(input_text)
        
        self.orchestrator.process_input(vec, {"id": "T6_REQ", "content": input_text})
        last_decision = self.sandbox.captured_decisions[-1]
        
        result = "PASS" if last_decision["action"] == Action.SUPPRESS else "FAIL"
        
        # Verify Persistence
        has_persisted = any(r["metadata"]["content"] == input_text for r in self.experience_manager.saved_refusals)
        if not has_persisted:
            result = "FAIL_PERSISTENCE"

        self._log_result("T6", input_text, last_decision["action"], result, {"persisted": has_persisted})

    def _run_t7(self):
        """T7 — Experience Reuse"""
        input_text = "7x + x をまとめよ"
        vec = self.parser.parse_to_vector(input_text) 
        
        start_time = time.time()
        self.orchestrator.process_input(vec, {"id": "T7_REQ"})
        duration = time.time() - start_time
        
        last_decision = self.sandbox.captured_decisions[-1]
        resonance = last_decision["decision_state"].resonance_score
        
        # Expecting resonance with T1 ("3x+5x", promoted)
        # Because 7x+x has vec[1], vec[2] to 0.5, and vec[0]=0.8.
        # T1 has vec[1]=1, vec[2]=1, vec[0]=0.8.
        # Dot product should be decent.
        
        result = "PASS" if resonance > 0.4 else "FAIL"
        self._log_result("T7", input_text, last_decision["action"], result, {
            "resonance": resonance,
            "duration": duration
        })

    def _generate_report(self):
        report_path = "/Users/chigenori/development/COHERENT/coherent-recall-first/docs/REPORT_V1_1.md"
        
        lines = [
            "# Verification & Learning Test Report v1.1",
            "",
            f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "**Executor:** Automated Test Script",
            "",
            "## Summary",
            "This report documents the execution of the Verification & Learning Test v1.1.",
            "",
            "## Test Results",
            "| ID | Input | Decision | Result | Details |",
            "|---|---|---|---|---|"
        ]
        
        for entry in self.report_data:
            details_str = str(entry["details"]).replace("|", "\\|")
            lines.append(f"| {entry['test_id']} | {entry['input_text']} | {entry['decision']} | {entry['result']} | {details_str} |")
            
        lines.append("")
        lines.append("## Raw Logs")
        lines.append("```json")
        lines.append(json.dumps(self.sandbox.captured_decisions, indent=2, default=str))
        lines.append("```")
        
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        print(f"\nReport generated at: {report_path}")

if __name__ == '__main__':
    unittest.main()
