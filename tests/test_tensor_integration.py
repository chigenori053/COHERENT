import pytest
import torch
from unittest.mock import MagicMock
from causalscript.core.reasoning.generator import HypothesisGenerator
from causalscript.core.reasoning.simulator import LookaheadSimulator
from causalscript.core.reasoning.types import Hypothesis
from causalscript.core.tensor.engine import TensorLogicEngine
from causalscript.core.tensor.converter import TensorConverter
from causalscript.core.tensor.embeddings import EmbeddingRegistry

class TestTensorIntegration:
    @pytest.fixture
    def setup_components(self):
        # Mocks
        registry = MagicMock()
        symbolic_engine = MagicMock()
        goal_scanner = MagicMock()
        
        # Real Tensor Components
        emb_registry = EmbeddingRegistry()
        converter = TensorConverter(emb_registry)
        tensor_engine = TensorLogicEngine(vocab_size=10, embedding_dim=4)
        
        return registry, symbolic_engine, goal_scanner, tensor_engine, converter

    def test_generator_integration(self, setup_components):
        registry, sym_engine, _, tensor_engine, converter = setup_components
        
        # Setup registry mock
        # generator calls match_rules -> returns list of (rule, next_expr)
        mock_rule = MagicMock()
        mock_rule.id = "rule_1"
        mock_rule.description = "Test Rules"
        mock_rule.category = "test"
        mock_rule.priority = 10
        
        registry.match_rules.return_value = [(mock_rule, "next_expr")]
        
        # Instantiate Generator with Tensor Engine
        generator = HypothesisGenerator(
            registry=registry,
            engine=sym_engine,
            tensor_engine=tensor_engine,
            tensor_converter=converter
        )
        
        # Register a rule in tensor engine
        tensor_engine.register_rule("rule_1")
        
        # Run Generate
        candidates = generator.generate("x = y")
        
        assert len(candidates) == 1
        assert candidates[0].rule_id == "rule_1"
        
        # Verification:
        # Since we didn't mock tensor_engine.predict_rules but used the real one,
        # checking if it crashed or not is the main verification of integration flow.
        # Also could inspect if predict_rules was called if we mocked it, but real is fine.

    def test_simulator_integration(self, setup_components):
        registry, _, goal_scanner, tensor_engine, converter = setup_components
        
        # Setup mocks
        goal_state = MagicMock()
        goal_state.complexity_score = 5.0
        goal_state.is_solved = False
        goal_scanner.scan.return_value = goal_state
        
        # Setup Generator
        generator = MagicMock()
        
        # Configure registry to return nodes with priority
        mock_node = MagicMock()
        mock_node.priority = 10.0
        # Determine how rules_by_id is accessed. Usually registry.rules_by_id.get()
        # So we mock rules_by_id dictionary behavior
        registry.rules_by_id = {"r1": mock_node, "r2": mock_node}
    
        # Simulator
        simulator = LookaheadSimulator(
            generator=generator,
            registry=registry,
            goal_scanner=goal_scanner,
            tensor_engine=tensor_engine,
            tensor_converter=converter
        )
        
        # Candidates
        c1 = Hypothesis(id="1", rule_id="r1", current_expr="cur", next_expr="A(x)", score=0.0)
        c2 = Hypothesis(id="2", rule_id="r2", current_expr="cur", next_expr="B(y)", score=0.0)
        
        # Run Simulate
        scored = simulator.simulate([c1, c2])
        
        assert len(scored) == 2
        # Check scores are not None/Zero (base score = 100 - 5 = 95)
        # Tensor score adds norm of random vector.
        assert scored[0].score > 90.0 
        assert scored[1].score > 90.0
        # If tensor integration works, score should be != 95.0 exactly (unless norm is 0)
        # With randn init, norm > 0 usually.
        assert scored[0].score != 95.0
