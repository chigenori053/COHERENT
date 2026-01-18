"""
Verification Script for Task Gate MVP
"""
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from coherent.core.task_gate import TaskGate, TaskType, RouteType, EscalationReason
from coherent.core.cognitive_core import CognitiveCore, DecisionType
from coherent.core.memory.experience_manager import ExperienceManager

def test_standalone_task_gate():
    print("--- Testing Standalone Task Gate ---")
    gate = TaskGate()
    
    test_cases = [
        # Transform check
        ("Translate to Japanese", TaskType.TRANSFORM, RouteType.FAST_PATH, EscalationReason.NONE),
        ("Convert to JSON format", TaskType.TRANSFORM, RouteType.FAST_PATH, EscalationReason.NONE),
        
        # Retrieval check
        ("What is the definition of AI?", TaskType.RETRIEVAL, RouteType.FAST_PATH, EscalationReason.NONE),
        ("Explain the term 'Singularity'", TaskType.RETRIEVAL, RouteType.FAST_PATH, EscalationReason.NONE),
        
        # Reasoning / High Complexity check
        ("Compare A and B and determine which is better based on ROI.", TaskType.REASONING, RouteType.FULL_PATH, EscalationReason.NONE),
        ("If X is true, then Y must be false, unless Z intervenes. However, suppose W.", TaskType.REASONING, RouteType.FULL_PATH, EscalationReason.NONE),
        ("Translate this extremely long document that contains many clauses, conditions, and specialized terminology, requiring careful consideration of context if possible.", TaskType.TRANSFORM, RouteType.FULL_PATH, EscalationReason.NONE),
        
        # --- Escalation Tests ---
        # Mixed Task (Translate + Why)
        ("Translate this sentence and explain why you chose those words.", TaskType.TRANSFORM, RouteType.FULL_PATH, EscalationReason.MIXED_TASK),
        # Subjective Output
        ("Summarize this nicely.", TaskType.TRANSFORM, RouteType.FULL_PATH, EscalationReason.SUBJECTIVE_OUTPUT),
        ("適切に翻訳して", TaskType.TRANSFORM, RouteType.FULL_PATH, EscalationReason.SUBJECTIVE_OUTPUT)
    ]
    
    failures = 0
    for text, expected_type, expected_route, expected_esc in test_cases:
        decision = gate.assess_task(text)
        print(f"Input: '{text}'")
        print(f"  -> Detected: {decision.task_type.name}, Route: {decision.route.name}, Score: {decision.complexity_score}, Esc: {decision.escalation_reason.name}")
        
        if decision.task_type != expected_type:
            print(f"  [FAIL] Expected Type {expected_type.name}, got {decision.task_type.name}")
            failures += 1
        elif decision.route != expected_route:
            print(f"  [FAIL] Expected Route {expected_route.name}, got {decision.route.name}")
            failures += 1
        elif decision.escalation_reason != expected_esc:
            print(f"  [FAIL] Expected Escalation {expected_esc.name}, got {decision.escalation_reason.name}")
            failures += 1
        else:
            print("  [PASS]")
    
    if failures == 0:
        print("Standalone Gate Test: SUCCESS\n")
    else:
        print(f"Standalone Gate Test: FAILED ({failures} errors)\n")

def test_integrated_cognitive_core():
    print("--- Testing Integrated Cognitive Core Flow ---")
    
    # Mock ExperienceManager
    class MockExperienceManager(ExperienceManager):
        def __init__(self): pass
        def log_experience(self, *args): pass
        
    core = CognitiveCore(experience_manager=MockExperienceManager())
    
    # Case 1: Fast Path
    print("Test Case 1: Fast Path Input")
    decision_fast = core.process_input("Translate 'Hello' to Japanese")
    trace_fast = core.current_trace
    
    # Verify Trace: specific Recall/Reasoning steps should be missing or minimal
    event_steps = [e.step for e in trace_fast.events]
    print(f"  Events: {event_steps}")
    
    if "Recall" not in event_steps and "Reasoning" not in event_steps:
        print("  [PASS] Recall/Reasoning skipped for Fast Path")
    else:
        print("  [FAIL] Fast Path did not skip heavy steps")
        
    if decision_fast.action == "FAST_EXECUTE":
        print("  [PASS] Decision Action is FAST_EXECUTE")
        print(f"  Result Reason: {decision_fast.reason}")
        if "[Fast Transform]" in decision_fast.reason or "Translation Result" in decision_fast.reason:
             print("  [PASS] FastExecutor Output detected")
        else:
             print("  [FAIL] FastExecutor Output MISSING")
    else:
        print(f"  [FAIL] Expected FAST_EXECUTE, got {decision_fast.action}")

    # Case 2: Full Path
    print("\nTest Case 2: Full Path Input")
    decision_full = core.process_input("Analyze the causal relationship between economic growth and environmental damage.")
    trace_full = core.current_trace
    
    event_steps_full = [e.step for e in trace_full.events]
    print(f"  Events: {event_steps_full}")
    
    if "Recall" in event_steps_full and "Reasoning" in event_steps_full:
        print("  [PASS] Recall/Reasoning present for Full Path")
    else:
        print("  [FAIL] Full Path missing heavy steps")


if __name__ == "__main__":
    test_standalone_task_gate()
    test_integrated_cognitive_core()
