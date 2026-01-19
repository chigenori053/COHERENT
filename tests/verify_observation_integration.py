import logging
import sys
# Configure logging to stdout to see ObservationCore output
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(name)s - %(levelname)s - %(message)s')

from coherent.core.cognitive_core import CognitiveCore
from coherent.core.memory.experience_manager import ExperienceManager

# Mock ExperienceManager as it might require DB/Files
class MockExperienceManager(ExperienceManager):
    def __init__(self):
        pass
    def log_experience(self, signal, meta):
        print(f"[MockExperience] Logged: {meta}")

def verify_integration():
    print("--- Starting ObservationCore Integration Verification ---")
    
    # 1. Initialize CognitiveCore
    print("1. Initializing CognitiveCore...")
    core = CognitiveCore(experience_manager=MockExperienceManager())
    
    # 2. Process Input (Normal)
    print("\n2. Processing Input: 'Hello World'")
    decision = core.process_input("Hello World")
    print(f"   Decision: {decision.decision_type.name}")
    
    # Check if ObservationCore logged something (visually in stdout)
    
    # 3. Process Input (Simulated High Entropy/Uncertainty)
    # We can't easily force entropy without mocking internal engines, 
    # but we can try an input that might trigger Review or different path.
    # Or just check that observation happens again.
    print("\n3. Processing Input: 'Calculate the meaning of life'")
    decision_2 = core.process_input("Calculate the meaning of life")
    print(f"   Decision: {decision_2.decision_type.name}")

    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    verify_integration()
