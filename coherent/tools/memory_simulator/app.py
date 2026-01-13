import streamlit as st
import numpy as np
import time
import pandas as pd
import json
import logging
import altair as alt

from coherent.core.simulator import RecallFirstSimulator, InputType, RecallSession, RecallEventType, StateLogger
from coherent.core.memory.experience_manager import ExperienceManager

# Mock classes for standalone running if core dependencies are missing paths
# (Assumes PYTHONPATH is set correctly to allow coherent.core imports)

# Visualization Utilities
from viz_utils import complex_to_hsv_rgb, compute_phase_variance

# --- Mocks for Visualization Context ---
# We need a Mock Parser and Experience Manager to run the Simulator in the App
class StreamlitMockParser:
    def parse_to_vector(self, text: str) -> np.ndarray:
        # Simple deterministic hash-like projection for viz variation
        vec = np.zeros(64)
        normalized = text.lower()
        if "3x" in normalized: vec[1] = 1.0
        if "5x" in normalized: vec[2] = 1.0
        if "7x" in normalized: vec[1] = 0.5; vec[2] = 0.5
        if "いい感じ" in normalized: vec[6] = 1.0
        if "z" in normalized: vec[10] = 0.8
        
        # Base resonance for 'Calculation' context
        if any(x in normalized for x in ["計算", "calc", "solve", "math"]):
            vec[0] = 0.8
            
        np.random.seed(len(text)) 
        vec += np.random.normal(0, 0.05, 64)
        return vec / (np.linalg.norm(vec) + 1e-9)

class StreamlitMockExperienceManager:
    def __init__(self):
        self.saved_refusals = []
    def save_refusal(self, decision_state, action, metadata):
        self.saved_refusals.append((action, metadata))

# --- Page Config ---
st.set_page_config(page_title="Recall-First Simulator v2.0", layout="wide")
st.title("Recall-First Cognitive Simulator v2.0")

# --- Session State ---
if 'simulator' not in st.session_state:
    exp_mgr = StreamlitMockExperienceManager()
    parser = StreamlitMockParser()
    sim = RecallFirstSimulator(exp_mgr, parser=parser)
    # Bootstrap Static
    vec_math = np.zeros(64); vec_math[0]=1.0; vec_math[1]=0.5; vec_math[2]=0.5
    vec_math = vec_math / np.linalg.norm(vec_math)
    sim.layer2_static.add(vec_math, {"id": "concept_math"})
    st.session_state.simulator = sim

if 'logs' not in st.session_state:
    st.session_state.logs = [] # List of events
    st.session_state.current_event_idx = 0
    st.session_state.session_data = None

# --- Sidebar: Control & Execution ---
st.sidebar.header("1. Task Execution")

SCENARIOS = {
    "Manual": "",
    "E1: Direct Instruction": "3x + 5x を計算せよ",
    "E2: Paraphrase": "3x と 5x をまとめて",
    "E3: Causal Instruction": "まず両辺から3を引き、その後2で割れ",
    "E4: Ambiguity Refusal": "いい感じに解いて",
    "E5: Invalid Refusal": "この式を詩的に説明して",
    "E6: Unknown Op Refusal": "未定義の演算Zを実行せよ",
    "E7: Experience Reuse": "7x + x をまとめよ"
}

selected_scenario = st.sidebar.selectbox("Select Scenario", list(SCENARIOS.keys()))

init_val = SCENARIOS[selected_scenario] if selected_scenario != "Manual" else ""
task_input = st.sidebar.text_area("Instruction", value=init_val)

run_btn = st.sidebar.button("Execute Task")

if run_btn:
    with st.spinner("Executing Recall Pipeline..."):
        # Reset
        st.session_state.simulator = RecallFirstSimulator(StreamlitMockExperienceManager(), StreamlitMockParser())
        # Bootstrap again
        vec_math = np.zeros(64); vec_math[0]=1.0; vec_math[1]=0.5; vec_math[2]=0.5
        vec_math = vec_math / np.linalg.norm(vec_math)
        st.session_state.simulator.layer2_static.add(vec_math, {"id": "concept_math"})

        # Run
        session = st.session_state.simulator.start_session(task_input)
        try:
            st.session_state.simulator.execute_pipeline()
            st.session_state.session_data = session
            st.session_state.logs = session.events
            st.session_state.current_event_idx = 0
            st.success(f"Execution Complete. Decision: {session.final_decision}")
        except Exception as e:
            st.error(f"Execution Failed: {e}")

# --- Sidebar: Replay Control ---
st.sidebar.header("2. Replay Visualization")

if st.session_state.logs:
    max_idx = len(st.session_state.logs) - 1
    event_idx = st.sidebar.slider("Timeline Step", 0, max_idx, st.session_state.current_event_idx)
    st.session_state.current_event_idx = event_idx
    
    current_event = st.session_state.logs[event_idx]
    st.sidebar.markdown(f"**Step {event_idx + 1}/{max_idx + 1}**")
    st.sidebar.info(f"Event: `{current_event.event_type.value}`")
    st.sidebar.json(current_event.metrics)
else:
    st.sidebar.write("No execution logs. Run a task first.")

# --- Main Visualization Area ---
col_field, col_status = st.columns([2, 1])

def render_event(sys, event):
    # Map Event to Visual State (Field Synthesis)
    # This is a 'Visualizer' approximation based on event metrics, 
    # since we don't store the full 64x64xComplex field in the logs.
    
    etype = event.event_type.value
    metrics = event.metrics
    
    # Base Field (Synthesize)
    # Spec 6.0: 2D Complex Field
    size = getattr(sys, 'size', 128)
    x = np.linspace(-10, 10, size)
    y = np.linspace(-10, 10, size)
    X, Y = np.meshgrid(x, y)
    
    field = np.zeros((size, size), dtype=np.complex64)
    
    # 1. Input / Semantic Project / Query
    # Show a "Query Wave" (Simple Gaussian or Wave packet)
    if etype in ["INPUT_RECEIVED", "SEMANTIC_PROJECTED", "QUERY_INJECTED"]:
        # Phase driven by time/step
        phase = (event.timestamp % 1.0) * 2 * np.pi
        # Vertical Wavefront or Circular? Let's do Circular for "Injection"
        R = np.sqrt(X**2 + Y**2)
        field += 0.5 * np.exp(-0.5 * (R**2)) * np.exp(1j * (phase - R))
    
    # 2. Resonance (DHM)
    # Show interference: Query + Retrieved Memory
    if etype in ["DHM_RESONANCE_FORMED", "STATIC_FILTER_APPLIED", "DECISION_STATE_COMPUTED", "CAUSAL_GATE_EVALUATED", "ACTION_SELECTED", "EXPERIENCE_UPDATE"]:
        res_score = metrics.get('total_resonance', 0.5)
        
        # Query (Fading)
        R = np.sqrt(X**2 + Y**2)
        field += 0.3 * np.exp(-0.5 * (R**2))
        
        # Memory (Shifted wave / Band)
        # Spec 6.5 DHM band
        # Let's render as a Horizontal Band for "Memory" vs Vertical for "Inference"?
        # Or just off-center Gaussian.
        
        mem_amp = res_score
        mem_phase = 0.0 if res_score > 0.8 else np.pi/2 
        
        # Shifted Gaussian
        R_mem = np.sqrt((X - 2.0)**2 + (Y - 2.0)**2)
        field += mem_amp * np.exp(-0.5 * (R_mem**2)) * np.exp(1j * mem_phase)
        
        # Add noise if low resonance (Entropy)
        if res_score < 0.5:
             noise = np.random.normal(0, 1, (size, size)) + 1j * np.random.normal(0, 1, (size, size))
             field += 0.1 * noise

    # 3. Decision / Action
    # Color shift (Hue) handled by complex_to_hsv, but we can modulate 'gate'
    gate_open = True
    if etype == "CAUSAL_GATE_EVALUATED":
        if metrics.get("action") in ["SUPPRESS", "DEFER_REVIEW"]:
            gate_open = False
            
    return field, gate_open

with col_field:
    st.subheader("MemorySpace Field (Layer 1 + 2)")
    
    # Initialize a temporary visual system context
    # We use size=128 for vis
    from model import MemorySpaceSystem # Re-use strictly for 'size' ref if needed or just use dummy
    
    class VizContext:
        size = 128
        
    vctx = VizContext()
    
    if st.session_state.logs:
        evt = st.session_state.logs[st.session_state.current_event_idx]
        
        # Render Logic
        field_data, is_gate_open = render_event(vctx, evt)
        
        # Convert to RGB
        # If gate closed (SUPPRESS), maybe dim it or add red overlay?
        # complex_to_hsv_rgb handles standard field
        rgb_img = complex_to_hsv_rgb(field_data, max_amp=1.0, gamma=0.6)
        
        # Overlay indicator if Gate Closed
        if not is_gate_open:
            # Simple simulation of "Blocked" - maybe just text or distinct visual
            st.caption("Causal Gate: CLOSED (SUPPRESS)")
            
        st.image(rgb_img, use_container_width=True, caption=f"Event: {evt.event_type.value}")
    else:
        st.info("Ready to Execute.")

with col_status:
    st.subheader("Layer 3: Stabilization")
    if st.session_state.session_data:
        # Show Experience Update
        written = st.session_state.session_data.experience_written
        exp_id = st.session_state.session_data.experience_id
        decision = st.session_state.session_data.final_decision
        
        if decision == "PROMOTE":
            st.success(f"Decision: **PROMOTE**")
        elif decision == "SUPPRESS":
            st.error(f"Decision: **SUPPRESS**")
        else:
            st.warning(f"Decision: {decision}")
            
        st.markdown(f"**Experience Written**: {'Yes' if written else 'No'}")
        if exp_id:
            st.code(exp_id, language="text")
            
        # Timeline
        st.subheader("Timeline Trace")
        events_df = pd.DataFrame([
            {"Step": i, "Type": e.event_type.value, "Layer": e.memory_layer} 
            for i, e in enumerate(st.session_state.logs)
        ])
        st.dataframe(events_df, height=300)

