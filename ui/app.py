"""
Coherent Streamlit UI - Integrated System Tester
"""

import streamlit as st
import sys
import os
import json
import time
import io
from contextlib import redirect_stdout
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Core Imports
from coherent.engine.symbolic_engine import SymbolicEngine
from coherent.engine.computation_engine import ComputationEngine
from coherent.engine.validation_engine import ValidationEngine
from coherent.engine.hint_engine import HintEngine, HintPersona
from coherent.engine.core_runtime import CoreRuntime
from coherent.engine.latex_formatter import LaTeXFormatter
from coherent.engine.parser import Parser
from coherent.engine.evaluator import Evaluator
from coherent.engine.learning_logger import LearningLogger
from coherent.engine.fuzzy.judge import FuzzyJudge
from coherent.engine.fuzzy.encoder import ExpressionEncoder
from coherent.engine.fuzzy.metric import SimilarityMetric
from coherent.engine.unit_engine import get_common_units
from coherent.engine.decision_theory import DecisionConfig
from coherent.engine.knowledge_registry import KnowledgeRegistry

# Reasoning & Memory Imports
from coherent.engine.tensor.engine import TensorLogicEngine
from coherent.engine.tensor.converter import TensorConverter
from coherent.engine.tensor.embeddings import EmbeddingRegistry
from coherent.engine.reasoning.agent import ReasoningAgent

import torch
import numpy as np
import matplotlib.pyplot as plt
import graphviz

# Page Config
st.set_page_config(
    page_title="Coherent System 2.0",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []
if "logs" not in st.session_state:
    st.session_state.logs = []
if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = []

# --- System Initialization ---

@st.cache_resource
def get_system():
    """Initializes the full Coherent System (Runtime + Agent + Memory)."""
    
    # 1. Base Engines
    sym_engine = SymbolicEngine()
    comp_engine = ComputationEngine(sym_engine)
    
    # 2. Knowledge Registry
    knowledge_path = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "coherent", "engine", "knowledge")))
    knowledge_path.mkdir(parents=True, exist_ok=True)
    knowledge_registry = KnowledgeRegistry(knowledge_path, sym_engine)
    
    # 3. Validation & Fuzzy Judge (Default Config)
    decision_config = DecisionConfig(strategy="balanced")
    encoder = ExpressionEncoder()
    metric = SimilarityMetric()
    fuzzy_judge = FuzzyJudge(encoder, metric, decision_config=decision_config, symbolic_engine=sym_engine)
    decision_engine = fuzzy_judge.decision_engine
    
    val_engine = ValidationEngine(
        comp_engine, 
        fuzzy_judge=fuzzy_judge,
        decision_engine=decision_engine,
        knowledge_registry=knowledge_registry
    )
    
    # 4. Hint Engine
    hint_engine = HintEngine(comp_engine)
    
    # 5. Core Runtime
    # Note: LearningLogger is typically per-session, but we need a base runtime for the Agent.
    # The Agent might use its own internal logger or the runtime's.
    logger = LearningLogger() 
    runtime = CoreRuntime(
        comp_engine, 
        val_engine, 
        hint_engine, 
        learning_logger=logger,
        knowledge_registry=knowledge_registry,
        decision_config=decision_config
    )
    
    # Inject units
    for name, unit in get_common_units().items():
        comp_engine.bind(name, unit)

    # 6. Tensor/Neuro-Symbolic Components
    embedding_registry = EmbeddingRegistry()
    tensor_converter = TensorConverter(embedding_registry)
    tensor_engine = TensorLogicEngine(vocab_size=1000, embedding_dim=64)
    
    # 7. Reasoning Agent (Recall-First)
    agent = ReasoningAgent(
        runtime,
        tensor_engine=tensor_engine,
        tensor_converter=tensor_converter
    )
    
    # 8. Utilities
    from coherent.engine.classifier import ExpressionClassifier
    classifier = ExpressionClassifier(sym_engine)
    formatter = LaTeXFormatter(sym_engine, classifier)

    return {
        "runtime": runtime,
        "agent": agent,
        "comp_engine": comp_engine,
        "knowledge_registry": knowledge_registry,
        "formatter": formatter,
        "tensor_engine": tensor_engine,
        "val_engine": val_engine,
        "hint_engine": hint_engine,
        "sym_engine": sym_engine
    }

# Load System
system = get_system()
runtime = system["runtime"]
agent = system["agent"]
formatter = system["formatter"]

# --- Helper Functions ---

def render_optical_memory(agent):
    """Visualizes the Optical Memory state as a heatmap."""
    try:
        # Access the optical memory tensor
        # Path: agent.generator.optical_layer.optical_memory (complex tensor) or similar
        # Based on file read: agent.trainer.model refers to generator.optical_layer
        optical_mem = agent.trainer.model.optical_memory # [Capacity, Dim]
        
        if optical_mem is None:
            return None
            
        # Get Magnitude (Energy) for visualization
        # Clone and detach to avoid interfering with gradients
        energy = torch.abs(optical_mem).detach().cpu().numpy()
        
        # We only show the first N slots that are non-zero/active to keep it readable
        # Or simple show top 50 rows
        display_rows = min(50, energy.shape[0])
        display_data = energy[:display_rows, :]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        cax = ax.imshow(display_data, aspect='auto', cmap='inferno', interpolation='nearest')
        ax.set_title(f"Optical Memory State (Top {display_rows} Slots)")
        ax.set_xlabel("Vector Dimension")
        ax.set_ylabel("Memory Slot")
        fig.colorbar(cax, orientation='vertical')
        
        return fig
    except Exception as e:
        st.error(f"Visualization Error: {e}")
        return None

# --- UI Layout ---

st.title("🌌 Coherent System 2.0")
st.markdown("Recall-First Reasoning & Optical Holographic Memory Interface")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Settings")
    
    strategy_name = st.selectbox(
        "Decision Strategy", 
        ["balanced", "strict", "encouraging"],
        index=0
    )
    
    # Update Runtime Config dynamically
    runtime.hint_persona = st.selectbox("Hint Persona", ["balanced", "sparta", "support"])
    runtime.validation_engine.fuzzy_judge.decision_engine.config.strategy = strategy_name
    
    st.divider()
    st.markdown("**System Status**")
    st.markdown(f"🧠 **Knowledge Rules**: {len(system['knowledge_registry'].nodes)}")
    # Assuming we can inspect memory capacity
    mem_cap = agent.trainer.model.memory_capacity
    st.markdown(f"💡 **Optical Capacity**: {mem_cap}")

# Tabs
tab_solver, tab_tester, tab_train = st.tabs(["🧩 Agent Solver", "✅ Script Tester", "🧠 Optical Training"])

# --- TAB 1: Agent Solver (Step-by-Step) ---
with tab_solver:
    st.header("Step-by-Step Reasoning")
    st.markdown("Solve problems using the **Reasoning Agent** (System 2) backed by **Optical Memory** (System 1).")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        problem_input = st.text_input("Enter Math Problem", value="(x - 2y)^2")
        solve_btn = st.button("Thinking Step", type="primary")
        
        reset_btn = st.button("Reset Problem")
        if reset_btn:
             st.session_state.agent_memory = []
             st.rerun()

    with col2:
        # Mini Visualization of Current State
        current_display = problem_input
        if st.session_state.agent_memory:
            current_display = st.session_state.agent_memory[-1]['state']
        
        st.info(f"Current State: `{current_display}`")

    if solve_btn:
        start_state = current_display
        
        # Capture Stdout for "Thought Trace"
        f = io.StringIO()
        with redirect_stdout(f):
            try:
                hypothesis = agent.think(start_state)
            except Exception as e:
                st.error(f"Agent Error: {e}")
                hypothesis = None
        
        thought_log = f.getvalue()
        
        if hypothesis:
            # Update Session State
            step_record = {
                "step": len(st.session_state.agent_memory) + 1,
                "input": start_state,
                "state": hypothesis.next_expr,
                "rule": hypothesis.rule_id,
                "score": hypothesis.score,
                "explanation": hypothesis.explanation,
                "log": thought_log
            }
            st.session_state.agent_memory.append(step_record)
        else:
            st.warning("Agent could not find a confident next step.")
            st.text(thought_log)

    # Display History / Solution Path
    if st.session_state.agent_memory:
        st.divider()
        st.subheader("Solution Path")
        
        for i, step in enumerate(st.session_state.agent_memory):
            with st.container():
                st.markdown(f"#### Step {i+1}")
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.latex(formatter.format_expression(step['state']))
                    st.caption(step['explanation'])
                with c2:
                    st.markdown(f"**Rule**: `{step['rule']}`")
                    st.markdown(f"**Confidence**: `{step['score']:.2f}`")
                    with st.expander("Machine Thoughts"):
                        st.code(step['log'], language="text")
                st.divider()

    # Optical Visualization
    with st.expander("👁️ Optical Memory State", expanded=True):
        fig = render_optical_memory(agent)
        if fig:
            st.pyplot(fig)


# --- TAB 2: Script Tester (Legacy/Batch) ---
with tab_tester:
    st.header("Validation Script Tester")
    
    default_script = """problem: (x - 2y)^2
step: (x - 2y)(x - 2y)
step: x(x - 2y) - 2y(x - 2y)
step: x^2 - 2xy - 2yx + 4y^2
step: x^2 - 4xy + 4y^2
end: x^2 - 4xy + 4y^2"""
    
    script_input = st.text_area("Validation Script", value=default_script, height=300)
    run_script = st.button("Validate Script", type="primary")
    
    if run_script:
        # Clear logs
        logger = system['runtime'].learning_logger
        logger.records = [] 
        
        try:
            parser = Parser(script_input)
            program = parser.parse()
            evaluator = Evaluator(program, system['runtime'], learning_logger=logger)
            success = evaluator.run()
            
            logs = logger.to_list()
            
            # Simple Render
            for record in logs:
                if record['phase'] == 'step':
                    status_icon = "✅" if record.get('status') == 'ok' else "❌"
                    st.markdown(f"{status_icon} **{record.get('expression')}**")
                    if record.get('status') != 'ok':
                        st.error(f"Status: {record.get('status')}")
                        if 'hint' in record.get('meta', {}):
                            st.info(f"Hint: {record['meta']['hint'].get('message')}")
                    
        except Exception as e:
            st.error(f"Error: {e}")


# --- TAB 3: Optical Training ---
with tab_train:
    st.header("Optical Layer Training")
    st.markdown("Train the agent's intuition (Optical Memory) from successful past experiences.")

    if st.button("Extract & Train from Logs"):
        # Use logs from Validation runs or System runs
        # We need a shared log store or merge them. 
        # For prototype, we use the Runtime's logger which Script Tester uses.
        
        logger = system['runtime'].learning_logger
        logs = logger.to_list()
        
        training_samples = []
        rule_ids = agent.generator.rule_ids # We need the mapping of rule_id to index
        
        for record in logs:
            if record.get('status') == 'ok' and record.get('phase') == 'step':
                expr = record.get('expression')
                meta = record.get('meta', {})
                rule_node = meta.get('rule', None) # Rule object
                
                if expr and rule_node:
                    rid = rule_node.get('id')
                    if rid and rid in rule_ids:
                        target_idx = rule_ids.index(rid)
                        training_samples.append((expr, target_idx))
        
        if training_samples:
            st.info(f"Found {len(training_samples)} samples.")
            with st.spinner("Training Optical Layer..."):
                loss = agent.retrain(training_samples, epochs=10)
            st.success(f"Training Complete. Loss: {loss:.4f}")
        else:
            st.warning("No valid training samples found in current session logs. Run a valid Validation Script first.")
