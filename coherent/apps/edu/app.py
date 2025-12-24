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
from coherent.core.symbolic_engine import SymbolicEngine
from coherent.core.computation_engine import ComputationEngine
from coherent.core.validation_engine import ValidationEngine
from coherent.core.hint_engine import HintEngine, HintPersona
from coherent.core.core_runtime import CoreRuntime
from coherent.core.latex_formatter import LaTeXFormatter
from coherent.core.parser import Parser
from coherent.core.evaluator import Evaluator
from coherent.core.learning_logger import LearningLogger
from coherent.core.fuzzy.judge import FuzzyJudge
from coherent.core.fuzzy.encoder import ExpressionEncoder
from coherent.core.fuzzy.metric import SimilarityMetric
from coherent.core.unit_engine import get_common_units
from coherent.core.decision_theory import DecisionConfig
from coherent.core.knowledge_registry import KnowledgeRegistry

# Reasoning & Memory Imports
from coherent.core.tensor.engine import TensorLogicEngine
from coherent.core.tensor.converter import TensorConverter
from coherent.core.tensor.embeddings import EmbeddingRegistry
from coherent.core.reasoning.agent import ReasoningAgent

# Language Processing
# Language Processing
from coherent.plugins.language.semantic_parser import RuleBasedSemanticParser
from coherent.plugins.language.semantic_types import TaskType
from coherent.plugins.language.decomposer import Decomposer

# Multimodal & UI
from PIL import Image
from coherent.core.multimodal.integrator import MultimodalIntegrator

# [NEW] Action Architecture
from coherent.core.action import Action
from coherent.core.action_types import ActionType
from coherent.core.state import State
from coherent.core.executor import ActionExecutor
from coherent.core.tracer import Tracer

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
if "tracer" not in st.session_state:
    st.session_state.tracer = Tracer()
if "lm_state" not in st.session_state:
    st.session_state.lm_state = None

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
    from coherent.core.classifier import ExpressionClassifier
    classifier = ExpressionClassifier(sym_engine)
    formatter = LaTeXFormatter(sym_engine, classifier)
    
    # 9. Language Parser
    semantic_parser = RuleBasedSemanticParser()

    return {
        "runtime": runtime,
        "agent": agent,
        "comp_engine": comp_engine,
        "knowledge_registry": knowledge_registry,
        "formatter": formatter,
        "tensor_engine": tensor_engine,
        "val_engine": val_engine,
        "hint_engine": hint_engine,
        "sym_engine": sym_engine,
        "semantic_parser": semantic_parser,
        "executor": ActionExecutor(runtime) # Stateless executor
    }


# Load System
system = get_system()
runtime = system["runtime"]
agent = system["agent"]
formatter = system["formatter"]
parser = system["semantic_parser"]
executor = system["executor"]

# --- Helper Functions ---

def render_optical_memory(agent):
    """Visualizes the Optical Memory state as a heatmap."""
    try:
        optical_mem = agent.trainer.model.optical_memory 
        
        if optical_mem is None:
            return None
            
        energy = torch.abs(optical_mem).detach().cpu().numpy()
        
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
    
    runtime.hint_persona = st.selectbox("Hint Persona", ["balanced", "sparta", "support"])
    runtime.validation_engine.fuzzy_judge.decision_engine.config.strategy = strategy_name
    
    st.divider()
    st.markdown("**System Status**")
    st.markdown(f"🧠 **Knowledge Rules**: {len(system['knowledge_registry'].nodes)}")
    mem_cap = agent.trainer.model.memory_capacity
    st.markdown(f"💡 **Optical Capacity**: {mem_cap}")

# Tabs
tab_solver, tab_tester, tab_train = st.tabs(["🧩 Agent Solver", "✅ Script Tester", "🧠 Optical Training"])

# --- TAB 1: Agent Solver (Step-by-Step) ---
# --- TAB 1: Chat Solver (System 2) ---
with tab_solver:
    st.header("Coherent Reasoning Agent")
    st.markdown("Interact with the **Reasoning LM** using natural language or images. Try 'Solve...' or verify 'A=B'.")

    # Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "image" in msg:
                st.image(msg["image"], width=200)

    # Input Area
    input_col1, input_col2 = st.columns([0.85, 0.15])
    
    with input_col1:
        prompt = st.chat_input("Ask a math question (e.g., 'Solve x^2-4=0') or verify a step ('(a+b)^2 = a^2+2ab+b^2')")
        
    with input_col2:
        uploaded_file = st.file_uploader("📷", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

    # Processing
    if prompt or uploaded_file:
        # User Message
        user_content = prompt if prompt else "Analyze this image."
        st.session_state.messages.append({"role": "user", "content": user_content})
        
        with st.chat_message("user"):
            st.markdown(user_content)
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, width=200)
                st.session_state.messages[-1]["image"] = image

        # Assistant Response
        with st.chat_message("assistant"):
            st.markdown("Thinking...")
            
            # 1. Multimodal Integration
            integrator = MultimodalIntegrator()
            expression = None
            if uploaded_file:
                # currently just using the text if provided, or default
                pass
            
            if prompt:
                 expression = prompt
            
            if expression:
                # 2. Decompose & Loop
                try:
                    # Initialize Tracer for this batch
                    if "tracer" not in st.session_state:
                         st.session_state.tracer = Tracer()
                    st.session_state.tracer.start_episode(expression)
                    
                    decomposer = Decomposer()
                    segments = decomposer.decompose(expression)
                    
                    if len(segments) > 1:
                        st.caption(f"🧩 Decomposed into {len(segments)} segments: {segments}")
                    
                    for i, segment in enumerate(segments):
                        st.markdown(f"**Segment {i+1}**: `{segment}`")
                        
                        # Parse
                        ir = parser.parse(segment)
                        
                        if not ir.inputs:
                            st.warning(f"Could not extract math from segment: {segment}")
                            continue
                            
                        target_expr = ir.inputs[0].value
                        
                        # Create State (New state for each segment, but Runtime persists)
                        state = State(
                            task_goal=ir.task,
                            initial_inputs=ir.inputs,
                            current_expression=target_expr
                        )
                        
                        # Agent Act
                        action = agent.act(state)
                        
                        # Execute
                        result = executor.execute(action, state)
                        
                        # Log & Render
                        st.session_state.tracer.log_step(state, action, result)
                        
                        with st.expander(f"Step {i+1} Result: {action.name}", expanded=True):
                            if "before" in result and "after" in result:
                                st.latex(f"{result['before']} \\rightarrow {result['after']}")
                            
                            if result.get("valid"):
                                st.success(f"✅ {action.inputs.get('explanation', 'Valid')}")
                                if ir.task == TaskType.VERIFY:
                                    st.caption("✨ Learned this verification!")
                            else:
                                msg = result.get('error', 'Unknown error')
                                st.error(f"❌ {msg}")
                                # Stop processing remaining segments on error?
                                # For now, continue to see full output or break?
                                # Better to break if it's a dependency chain.
                                st.error("Stopping execution due to error.")
                                break
                                
                    st.session_state.messages.append({"role": "assistant", "content": "Processed all segments."})

                except Exception as e:
                    st.error(f"Error processing request: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
            else:
                 st.info("Please provide text input or wait for OCR implementation.")



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
        logger = system['runtime'].learning_logger
        logger.records = [] 
        
        try:
            parser_v = Parser(script_input)
            program = parser_v.parse()
            evaluator = Evaluator(program, system['runtime'], learning_logger=logger)
            success = evaluator.run()
            
            logs = logger.to_list()
            
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
        logger = system['runtime'].learning_logger
        logs = logger.to_list()
        
        training_samples = []
        rule_ids = agent.generator.rule_ids
        
        for record in logs:
            if record.get('status') == 'ok' and record.get('phase') == 'step':
                expr = record.get('expression')
                meta = record.get('meta', {})
                rule_node = meta.get('rule', None) 
                
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
