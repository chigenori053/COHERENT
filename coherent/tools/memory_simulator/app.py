import streamlit as st
import numpy as np
import pandas as pd
import json
import altair as alt
import uuid
import sys
import os
from dataclasses import asdict

# Add Project Root to Path to resolve 'coherent' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Import Core (v2.0)
from coherent.core.cognitive_core import CognitiveCore, CognitiveDecision, CognitiveStateVector, DecisionType
from coherent.core.memory.experience_manager import ExperienceManager

# Visualization Utilities
from viz_utils import complex_to_hsv_rgb

# --- MOCKS (For Standalone Run) ---
class StreamlitMockExperienceManager:
    def __init__(self):
        self.saved_refusals = []
    def save_refusal(self, decision_state, action, metadata):
        # Adapter to match legacy call signature if needed, or update core to use simple log
        pass
    def log_experience(self, input_signal, meta):
        pass

# --- ADAPTER ---
# Convert CognitiveTrace events to format expected by Viz (or update Viz)
# We update Viz to work with CognitiveEvent

# --- CONFIG & STATE ---
st.set_page_config(page_title="BrainModel v2.0 Simulator", layout="wide")
st.title("BrainModel v2.0: Cognitive Core Simulator")

if 'cognitive_core' not in st.session_state:
    exp_mgr = StreamlitMockExperienceManager()
    # CognitiveCore handles its own internal memory initialization
    core = CognitiveCore(exp_mgr)
    
    # Bootstrap Memory (Directly accessing internal engine for demo)
    vec_math = np.zeros(64); vec_math[0]=1.0; vec_math[1]=0.5; vec_math[2]=0.5
    vec_math = vec_math / np.linalg.norm(vec_math)
    core.recall_engine.add(vec_math, {"id": "concept_algebra_basic", "content": "Basic Algebra Rules"})
    
    st.session_state.cognitive_core = core

if 'current_trace' not in st.session_state:
    st.session_state.current_trace = None # CognitiveTrace
if 'current_event_idx' not in st.session_state:
    st.session_state.current_event_idx = 0

# --- SIDEBAR: CONTROLS ---
st.sidebar.header("1. Execution Control")

# --- SETTINGS ---
with st.sidebar.expander("Settings", expanded=True):
    enable_simulation = st.checkbox("Enable SimulationCore", value=True, help="If unchecked, Logic Layer (Simulation) will be skipped even if triggered.")

# --- INPUT ---
st.sidebar.header("1. Execution Control")

SCENARIOS = {
    "Manual": "",
    "S1: Novelty (Clear Winner)": "3x + 5x",
    "S2: Conflict (Ambiguity)": "Concept A vs Concept B", 
    "S3: Simulation Trigger": "Solve 3.14 * 5.0"
}

selected_scenario = st.sidebar.selectbox("Select Scenario", list(SCENARIOS.keys()))
init_val = SCENARIOS[selected_scenario] if selected_scenario != "Manual" else ""

input_mode = st.sidebar.radio("Input Mode", ["Text Only", "Multimodal (Text + Image)"])

task_input_text = st.sidebar.text_area("Input Text (Signal)", value=init_val)
task_input_image = None

if input_mode == "Multimodal (Text + Image)":
    task_input_image = st.sidebar.file_uploader("Upload Context Image", type=["png", "jpg", "jpeg"])
    if task_input_image:
        st.sidebar.image(task_input_image, caption="Uploaded Config", use_container_width=True)

# Construct Signal
if task_input_image:
    # Save uploaded file to temp path so Core can read it (Strict Mode)
    upload_dir = os.path.join(os.path.dirname(__file__), "temp_uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, task_input_image.name)
    with open(file_path, "wb") as f:
        f.write(task_input_image.getbuffer())
        
    task_input = {
        "text": task_input_text,
        "image_name": os.path.abspath(file_path), # Pass absolute path
        # In real app, we would process bytes here
    }
else:
    task_input = task_input_text

if st.sidebar.button("Process Input"):
    with st.spinner("Thinking..."):
        # Reset Core for clean demo state if desired (optional)
        # st.session_state.cognitive_core = ... 
        
        # Run
        config = {"enable_simulation": enable_simulation}
        decision = st.session_state.cognitive_core.process_input(task_input, context_config=config)
        
        # Store Trace
        st.session_state.current_trace = st.session_state.cognitive_core.current_trace
        st.session_state.current_event_idx = 0
        
        st.success(f"Decision: {decision.decision_type.value}")

# --- RESULT SECTION ---
if st.session_state.current_trace:
    st.divider()
    res_col1, res_col2 = st.columns([3, 1])
    
    decision = st.session_state.current_trace.final_decision
    
    with res_col1:
        st.subheader("Final Decision")
        if decision:
            color = "green" if decision.decision_type == DecisionType.ACCEPT else "orange" if decision.decision_type == DecisionType.REVIEW else "red"
            st.markdown(f":{color}[**{decision.decision_type.value}**] - {decision.action}")
            st.info(f"Reason: {decision.reason}")
        else:
            st.warning("No decision recorded.")
            
    with res_col2:
        st.subheader("State Snapshot")
        if decision:
             st.metric("Entropy (H)", f"{decision.state_snapshot.entropy:.2f}")
             st.metric("Confidence (C)", f"{decision.state_snapshot.confidence:.2f}")
             st.metric("Reliability (R)", f"{decision.state_snapshot.recall_reliability:.2f}")

# --- MAIN: A-AXIS (TIMELINE) ---
st.header("Process Timeline (Trace)")

if st.session_state.current_trace and st.session_state.current_trace.events:
    events = st.session_state.current_trace.events
    
    # 1. Prepare Data for Altair
    # Map Steps to meaningful Y-axis
    step_order = ["Recall", "Reasoning", "Decision", "Simulation", "Experience", "Observation"]
    
    timeline_data = []
    for i, evt in enumerate(events):
        timeline_data.append({
            "step_id": i,
            "stage": evt.step, # Y-axis
            "description": evt.description,
            "timestamp": evt.timestamp,
            "selected": (i == st.session_state.current_event_idx)
        })
    
    df_timeline = pd.DataFrame(timeline_data)
    
    base = alt.Chart(df_timeline).encode(
        x=alt.X('step_id:O', title="Step Sequence"),
        y=alt.Y('stage:N', sort=step_order, title="Cognitive Stage"),
        tooltip=['stage', 'description']
    )
    
    points = base.mark_circle(size=300).encode(
        color=alt.condition(
            alt.datum.selected, 
            alt.value('red'),  # Selected
            alt.value('steelblue') # Default
        )
    )
    
    text = base.mark_text(dy=-20, size=12).encode(text='description')
    
    chart = (points + text).properties(height=250, width='container')
    st.altair_chart(chart, use_container_width=True)
    
    # 3. Control Slider
    max_idx = max(0, len(events) - 1)
    
    # Ensure session state index is valid for new max
    if st.session_state.current_event_idx > max_idx:
        st.session_state.current_event_idx = max_idx
        
    selected_idx = 0
    if max_idx > 0:
        selected_idx = st.slider("Select Event to Inspect", 0, max_idx, st.session_state.current_event_idx)
    else:
        st.caption("Single event trace (Navigation disabled)")
        
    st.session_state.current_event_idx = selected_idx
    
    current_event = events[selected_idx]
    
else:
    st.info("Process input to generate a Cognitive Trace.")
    current_event = None

# --- MAIN: B-AXIS (HOLOGRAPHIC & DETAILS) ---
st.divider()
st.header("internal State Inspection")

if current_event:
    col_vis, col_meta = st.columns([1, 1])
    
    with col_vis:
        st.subheader(f"Step: {current_event.step}")
        st.markdown(f"**{current_event.description}**")
        
        # Visualize "Field" conceptually based on Metrics
        # Simple dynamic field synthesis based on entropy/confidence
        
        size = 128
        field = np.zeros((size, size), dtype=np.complex64)
        x = np.linspace(-10, 10, size)
        y = np.linspace(-10, 10, size)
        X, Y = np.meshgrid(x, y)
        
        # Field Generation Logic
        if current_event.step == "Recall":
            # Show Resonance - multiple peaks?
            metrics = current_event.metrics
            top_score = metrics.get('top_score', 0)
            count = metrics.get('count', 0)
            
            # Central peak scaled by top_score
            R = np.sqrt(X**2 + Y**2)
            field += top_score * np.exp(-0.5 * R**2) * np.exp(1j * R)
            
            # Noise/Scatter if count > 1
            if count > 1:
                field += 0.2 * np.sin(X) * np.cos(Y)
                
        elif current_event.step == "Reasoning":
             # Show Network/Hypothesis formation
             # Interference pattern
             R1 = np.sqrt((X-2)**2 + (Y-2)**2)
             R2 = np.sqrt((X+2)**2 + (Y+2)**2)
             field += 0.5 * np.exp(-0.3*R1**2) + 0.5 * np.exp(-0.3*R2**2) * np.exp(1j*np.pi/2)
             
        elif current_event.step == "Decision":
             # Stability field
             entropy = current_event.metrics.get('entropy', 1.0)
             # Low entropy = Stable (Gaussian). High entropy = Turbulent/Flat.
             R = np.sqrt(X**2 + Y**2)
             width = 0.2 + (entropy * 2.0) # Wider if execution unclear
             field += 1.0 * np.exp(-width * R**2)
             
        elif current_event.step == "Observation":
             # Observation Field: "Eye" or "Scanner" metaphor
             # Ring shape
             R = np.sqrt(X**2 + Y**2)
             states = current_event.metrics.get("states", [])
             
             # Base Ring
             field += 0.8 * np.exp(-2.0 * (R - 5.0)**2) 
             
             # If Unstable / Uncertainty, add noise
             if "HIGH_UNCERTAINTY" in states:
                 field += 0.3 * np.random.randn(size, size)
                 
             # If Stable, add central clear point
             if "STRUCTURALLY_STABLE" in states:
                 field += 1.0 * np.exp(-0.5 * R**2) * 1j
             
        # Draw
        rgb = complex_to_hsv_rgb(field)
        st.image(rgb, caption="Cognitive Field Visualization", use_container_width=True)
        
    with col_meta:
        st.subheader("Detailed Metrics")
        st.json(current_event.metrics)
        
        if current_event.details:
            st.subheader("Context Details")
            st.json(current_event.details)
            
            # Special Visualization for Hypotheses
            if "hypotheses" in current_event.details:
                hyps = current_event.details['hypotheses']
                if hyps:
                    df = pd.DataFrame(hyps)
                    # Show bar chart of hypothesis scores
                    chart = alt.Chart(df).mark_bar().encode(
                        x='score:Q',
                        y=alt.Y('content:N', sort='-x'),
                        color='source:N',
                        tooltip=['content', 'score', 'source']
                    ).properties(title="Hypothesis Competition")
                    st.altair_chart(chart, use_container_width=True)
        
    # --- LOG EXPORT ---
    st.divider()
    with st.expander("Raw Execution Log (JSON)"):
        # Convert CognitiveTrace to dict 
        # (Naive serialization, in prod use default=str or similar)
        try:
            log_data = asdict(st.session_state.current_trace)
            st.json(log_data)
        except Exception as e:
            st.error(f"Serialization Error: {e}")
            st.write(st.session_state.current_trace)

else:
    st.write("Waiting for trace selection...")

