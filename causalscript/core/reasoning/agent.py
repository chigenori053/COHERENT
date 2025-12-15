from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List, Tuple

from .generator import HypothesisGenerator
from .goal import GoalScanner
from .simulator import LookaheadSimulator
from .types import Hypothesis
# [NEW] Import Trainer
from ..optical.trainer import OpticalTrainer

if TYPE_CHECKING:
    from ..core_runtime import CoreRuntime

class ReasoningAgent:
    """
    Autonomous Reasoning Agent (System 2) for CausalScript.
    Loops through Generate -> Simulate -> Evaluate to find the best next step.
    Supports Online Learning via OpticalTrainer.
    """

    def __init__(self, runtime: CoreRuntime, tensor_engine=None, tensor_converter=None):
        self.runtime = runtime
        if not self.runtime.knowledge_registry:
            raise ValueError("ReasoningAgent requires a KnowledgeRegistry")
            
        self.registry = self.runtime.knowledge_registry
        self.symbolic_engine = self.runtime.computation_engine.symbolic_engine
        
        self.generator = HypothesisGenerator(
            self.registry, 
            self.symbolic_engine,
            optical_weights_path=None 
        )
        self.goal_scanner = GoalScanner(self.symbolic_engine)
        self.simulator = LookaheadSimulator(self.generator, self.registry, self.goal_scanner)

        # [NEW] Initialize Trainer linked to the Generator's Optical Layer
        self.trainer = OpticalTrainer(
            model=self.generator.optical_layer,
            vectorizer=self.generator.vectorizer
        )

    def think(self, current_expr: str) -> Optional[Hypothesis]:
        """
        Executes one cycle of reasoning to find the best next step.
        """
        # 1. Generate: Find candidate rules
        candidates = self.generator.generate(current_expr)
        
        if not candidates:
            return None
            
        # 2. Simulate & Evaluate: Lookahead and scoring
        scored_candidates = self.simulator.simulate(candidates, depth=1)
        
        if not scored_candidates:
            return None
            
        # 3. Decision: Select best move
        best_move = max(scored_candidates, key=lambda x: x.score)
        
        self._add_explanation(best_move)
        
        return best_move

    def retrain(self, training_data: List[Tuple[str, int]], epochs: int = 1) -> float:
        """
        Triggers a retraining session for the Optical Layer.
        
        Args:
            training_data: List of (expression, target_rule_idx) tuples.
            epochs: Number of epochs to train.
            
        Returns:
            avg_loss: The average loss over the last epoch.
        """
        print(f"Starting Optical Retraining with {len(training_data)} samples...")
        avg_loss = 0.0
        for epoch in range(epochs):
            loss = self.trainer.train_epoch(training_data)
            avg_loss = loss
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")
            
        return avg_loss

    def _add_explanation(self, hypothesis: Hypothesis) -> None:
        """Generates a natural language explanation for the hypothesis."""
        rule_desc = hypothesis.metadata.get("rule_description", "Use a rule")
        hypothesis.explanation = f"{rule_desc} applied to transform the expression."
