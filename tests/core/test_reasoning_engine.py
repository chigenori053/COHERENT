
import unittest
import numpy as np
from coherent.core.reasoning_engine import ReasoningEngine, Hypothesis

class TestReasoningEngine(unittest.TestCase):
    def setUp(self):
        self.engine = ReasoningEngine()
        self.dummy_vec = np.zeros(64)

    def test_pass_through(self):
        """Verify strong recall results are passed through as Hypotheses"""
        # Mock Recall Results: [(Content, Score)]
        recall = [("Fact A", 0.9), ("Fact B", 0.4)]
        
        hypotheses = self.engine.generate_hypotheses(recall, self.dummy_vec)
        
        self.assertEqual(len(hypotheses), 2)
        self.assertEqual(hypotheses[0].content, "Fact A")
        self.assertEqual(hypotheses[0].score, 0.9)
        self.assertEqual(hypotheses[0].source, "Recall")

    def test_abduction_trigger(self):
        """Verify abductive reasoning (Novelty Hypothesis) triggers on low scores"""
        # All scores low (< 0.3)
        recall = [("Weak Fact", 0.2), ("Noise", 0.1)]
        
        hypotheses = self.engine.generate_hypotheses(recall, self.dummy_vec)
        
        # Should have original 2 + 1 Abductive
        self.assertEqual(len(hypotheses), 3)
        
        # Check for Abductive Hypothesis
        abductive = [h for h in hypotheses if h.source == "Abduction"]
        self.assertEqual(len(abductive), 1)
        self.assertEqual(abductive[0].score, 0.5)
        self.assertIn("Novelty", str(abductive[0].content))

    def test_empty_recall(self):
        """Verify empty recall returns empty list (CognitiveCore handles entropy)"""
        hypotheses = self.engine.generate_hypotheses([], self.dummy_vec)
        self.assertEqual(len(hypotheses), 0)

if __name__ == '__main__':
    unittest.main()
