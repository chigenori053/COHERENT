
import sys
import os
import importlib.util

def load_task_gate_module():
    # Load task_gate.py directly to bypass coherent/__init__.py and heavy dependencies
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../coherent/core/task_gate.py"))
    spec = importlib.util.spec_from_file_location("task_gate_module", file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["task_gate_module"] = module
    spec.loader.exec_module(module)
    return module

def verify_light_tasks():
    print("--- Verifying Light Tasks (Fast Path) ---")
    
    # Load module directly
    tg_module = load_task_gate_module()
    TaskGate = tg_module.TaskGate
    TaskType = tg_module.TaskType
    RouteType = tg_module.RouteType
    EscalationReason = tg_module.EscalationReason
    
    gate = TaskGate()
    
    # Define tasks that SHOULD be "Light" (Fast Path)
    light_task_candidates = [
        "Translate 'Hello world' to Japanese",
        "What is the definition of API?",
        "Summarize this text in 3 lines.",
        "Convert this number to JSON format.",
        "意味を教えて: 'Agentic'",
        "Paraphrase this sentence."
    ]
    
    # Define tasks that SHOULD be "Heavy" (Full Path)
    heavy_task_candidates = [
        "Explain why the sky is blue.",
        "Compare Python and Java.",
        "Evaluate this code nicely.", # "Nicely" is subjective -> Escalation
        "Translate this and explain the nuances." # Mixed -> Escalation
    ]
    
    passed_count = 0
    total_checks = 0
    
    print("\n[Light Task Candidates]")
    for text in light_task_candidates:
        total_checks += 1
        decision = gate.assess_task(text)
        
        print(f"Input: '{text}'")
        # print(f"  Result: {decision}")
        
        is_fast_path = (decision.route == RouteType.FAST_PATH)
        is_correct_type = (decision.task_type in [TaskType.TRANSFORM, TaskType.RETRIEVAL])
        is_low_complexity = (decision.complexity_score < gate.complexity_threshold)
        is_no_escalation = (decision.escalation_reason == EscalationReason.NONE)
        
        if is_fast_path and is_correct_type and is_low_complexity and is_no_escalation:
            print("  [PASS] -> Light Task (Fast Path)")
            passed_count += 1
        else:
            print("  [FAIL] -> NOT Light Task")
            print(f"     Type: {decision.task_type.name}, Route: {decision.route.name}, Esc: {decision.escalation_reason.name}")

    print("\n[Heavy/Escalated Task Candidates (Control Group)]")
    for text in heavy_task_candidates:
         total_checks += 1
         decision = gate.assess_task(text)
         print(f"Input: '{text}'")
         if decision.route == RouteType.FULL_PATH:
             print("  [PASS] -> Full Path (Correctly Blocked)")
             passed_count += 1
         else:
             print("  [FAIL] -> Should be Full Path but got Fast Path")

    print(f"\nSummary: {passed_count}/{total_checks} checks passed.")

if __name__ == "__main__":
    verify_light_tasks()
