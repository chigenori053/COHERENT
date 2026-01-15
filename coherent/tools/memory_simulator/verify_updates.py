import sys
import os
import uuid
import datetime
# Add Project Root to Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from coherent.core.simulator import RecallFirstSimulator, RecallSession, RecallEventType, InputType
from coherent.core.memory.experience_manager import ExperienceManager

# Mock ExperienceManager
class MockExpManager:
    def save_refusal(self, *args): pass

def verify_simulator_update():
    print("--- Verifying Reader Simulator Update ---")
    sim = RecallFirstSimulator(MockExpManager())
    
    # 1. Test Logic Computation
    print("\n[Test 1] Logic Computation Trigger")
    session = sim.start_session("3*x + 5*x", input_type=InputType.TEXT)
    sim.execute_pipeline()
    
    print(f"Result: {session.execution_result}")
    print(f"Source: {session.inference_source}")
    
    if session.execution_result == "8*x" and "Logic" in session.inference_source:
        print("PASS")
    else:
        print("FAIL")

    # 2. Test File Mock (Complex Input)
    print("\n[Test 2] File Input Handling")
    # Simulate file content passed via start_session
    file_content = "10 + 20"
    session2 = sim.start_session(file_content, input_type=InputType.FILE)
    sim.execute_pipeline()
    
    print(f"Result: {session2.execution_result}")
    print(f"Source: {session2.inference_source}")
    
    if session2.execution_result == "30" and session2.task.input_source.type == InputType.FILE:
        print("PASS")
    else:
        print("FAIL: Expected 30 and FILE type")

if __name__ == "__main__":
    verify_simulator_update()
