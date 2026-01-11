
import logging
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from coherent.core.cortex.controller import CortexController
from coherent.core.logic.controller import LogicController, LogicTask
from coherent.core.bridge import InterchangeData

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("C2_Generalization")

def run_experiment():
    print("=== C2 Generalization Experiment: Mathematical Pattern Discovery ===\n")

    # 1. Initialize Controllers
    logger.info("Initializing Cortex (Right Brain) and Logic (Left Brain)...")
    cortex = CortexController()
    logic = LogicController()
    
    # 2. Observation Phase (Cortex)
    observations = [
        "1 + 1 = 2",
        "2 + 2 = 4",
        "3 + 3 = 6"
    ]
    print(f"DEBUG: Cortex observes: {observations}")
    
    # 3. Hypothesis Generation (Cortex Abstraction)
    logger.info("Cortex is abstracting patterns from observations...")
    hypothesis: InterchangeData = cortex.propose_hypothesis(observations)
    
    if not hypothesis:
        logger.error("Cortex failed to generate a hypothesis.")
        return

    print(f"\n[Cortex Output]")
    print(f"  Hypothesis: {hypothesis.abstract_pattern}")
    print(f"  Rationale : {hypothesis.rationale}")
    print(f"  Confidence: {hypothesis.resonance_score}")
    
    # 4. Verification Request (Bridge -> Logic)
    logger.info("Sending hypothesis to Logic Orchestrator for formal verification...")
    
    # Create Logic Task
    # We want to verify if the pattern "x + x == 2*x" is universally true.
    # The hypothesis content is a string equation.
    # Logic Orchestrator should route this to Symbolic Engine for proof.
    task = LogicTask(
        task_type="verify",
        content={
            "before": "x + x", # Extracting LHS
            "after": "2 * x"   # Extracting RHS
            # Note: real C2 logic would parse "x+x==2*x" into LHS/RHS or pass as relation
        },
        context={},
        engine_hint="validation" # Explicitly asking for validation/proof
    )
    
    # Parsing helper for the demo (since hypothesis is raw string)
    if "==" in hypothesis.abstract_pattern:
        lhs, rhs = hypothesis.abstract_pattern.split("==")
        task.content = {"before": lhs.strip(), "after": rhs.strip()}
    
    # 5. Execution (Logic)
    result = logic.orchestrate(task)
    
    print(f"\n[Logic Output]")
    print(f"  Result: {result}")
    
    # 6. Consolidation (Feedback Loop)
    if isinstance(result, dict) and result.get("valid"):
        print(f"\nSUCCESS: Hypothesis '{hypothesis.abstract_pattern}' is formally PROVEN.")
        print("Cortex consolidates this rule into Long-Term Memory.")
    else:
        print(f"\nFAILURE: Hypothesis rejected.")

if __name__ == "__main__":
    run_experiment()
