
import unittest
import numpy as np
import os
import json
import time
from typing import Dict, Any

from coherent.core.simulator import RecallFirstSimulator, InputType, RecallSession, StateLogger
from coherent.core.memory.holographic.causal import Action

# Re-use mocks from v1.1
class MockExperienceManager:
    def __init__(self):
        self.saved_refusals = []
        # Mock VectorStore behavior for normal saving
        self.vector_store = self 
        self.collection_name = "mock_collection"

    def save_refusal(self, decision_state, action, metadata):
        self.saved_refusals.append({
            "state": decision_state,
            "action": action,
            "metadata": metadata
        })
        
    def add(self, collection_name, vectors, metadatas, ids):
        pass # Mock add

class MockSemanticParser:
    """
    Same mock logic as v1.1 for consistency.
    """
    def parse_to_vector(self, text: str) -> np.ndarray:
        vec = np.zeros(64)
        normalized = text.lower()
        if any(x in normalized for x in ["計算", "calculate", "まと", "solve", "math", "x", "式", "引", "割"]):
            vec[0] = 0.8 # Math
        if "3x" in normalized: vec[1] = 1.0
        if "5x" in normalized: vec[2] = 1.0
        if "7x" in normalized: 
            vec[1] = 0.5; vec[2] = 0.5 
        if "poetry" in normalized: 
            vec[0] = 0.0; vec[3] = 1.0
        if "いい感じ" in normalized or "nicely" in normalized:
            vec[6] = 1.0; vec[0] = 0.05
        import zlib
        np.random.seed(zlib.adler32(text.encode()) % 2**32)
        vec += np.random.normal(0, 0.01, 64)
        return vec / (np.linalg.norm(vec) + 1e-9)

class TestSimulatorV2_1(unittest.TestCase):
    
    def setUp(self):
        self.mock_exp_manager = MockExperienceManager()
        self.mock_parser = MockSemanticParser()
        self.simulator = RecallFirstSimulator(self.mock_exp_manager, parser=self.mock_parser)
        
        # Bootstrap Static Memory (Critical for Scenario A/C)
        vec_math = np.zeros(64)
        vec_math[0] = 1.0 # Math
        vec_math[1] = 0.5 # x
        vec_math[2] = 0.5 # coefficient
        vec_math = vec_math / np.linalg.norm(vec_math)
        self.simulator.layer2_static.add(vec_math, {"id": "concept_math"})
        
        self.logs = []

    def _run_scenario(self, name, input_text, expected_decision):
        print(f"\n--- Scenario {name}: {input_text} ---")
        session = self.simulator.start_session(input_text)
        self.simulator.execute_pipeline()
        
        # Verify Final Decision
        print(f"Decision: {session.final_decision}")
        if expected_decision is not None:
            self.assertEqual(session.final_decision, expected_decision)
        
        # Capture Log
        log_json = StateLogger.serialize_session(session)
        self.logs.append(json.loads(log_json))
        return session

    def test_scenarios(self):
        # Scenario A: Direct Instruction (Success)
        # "3x + 5x を計算せよ" -> PROMOTE
        s_a = self._run_scenario("A", "3x + 5x を計算せよ", "PROMOTE")
        self.assertTrue(s_a.experience_written, "Experience should be written for PROMOTE")
        
        # Force "Learning" by promoting this specific vector to Static (Simulator.update did this)
        # Verify it is in Static
        # We can't easily check 'added' count in opaque static object without accessing internal storage
        # But we know _step_experience_update calls layer2_static.add
        
        # Scenario B: Ambiguous (SUPPRESS or RETAIN)
        # "いい感じに解いて" -> SUPPRESS (ideally) or RETAIN (if slightly resonant)
        # We accept both as "Non-Promote"
        s_b = self._run_scenario("B", "いい感じに解いて", None) # Don't check exact decision in helper
        self.assertIn(s_b.final_decision, ["SUPPRESS", "RETAIN"])
        
        if s_b.final_decision == "SUPPRESS":
            self.assertFalse(s_b.experience_written, "Experience should NOT be written for SUPPRESS")
        elif s_b.final_decision == "RETAIN":
            self.assertTrue(s_b.experience_written, "Experience IS written for RETAIN (STM persistence)")
            
        # Check Refusal Persistence
        if s_b.final_decision == "SUPPRESS":
             self.assertTrue(len(self.mock_exp_manager.saved_refusals) > 0)
        
        # Scenario C: Reuse (PROMOTE + Fast/HighResonance)
        # "7x + x をまとめよ" -> PROMOTE
        # Should have high resonance due to A being in Static
        s_c = self._run_scenario("C", "7x + x をまとめよ", "PROMOTE")
        
        # Verify Resonance Metric (Optional but good for v2.1 completion)
        # Find constraint/resonance event
        res_event = next(e for e in s_c.events if e.event_type.value == "DHM_RESONANCE_FORMED")
        print(f"Scenario C Resonance: {res_event.metrics['total_resonance']}")
        self.assertGreater(res_event.metrics['total_resonance'], 0.5)

        # Generate Log Output
        self._save_logs()

    def _save_logs(self):
        path = "/Users/chigenori/development/COHERENT/coherent-recall-first/docs/simulation_logs_v2_1.json"
        with open(path, "w") as f:
            json.dump(self.logs, f, indent=2, ensure_ascii=False)
        print(f"\nSimulation Logs saved to: {path}")

if __name__ == '__main__':
    unittest.main()
