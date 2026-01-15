import streamlit as st
import numpy as np
import pandas as pd
import json
import altair as alt
import uuid
import sys
import os

# Add Project Root to Path to resolve 'coherent' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Import Core
from coherent.core.simulator import RecallFirstSimulator, RecallSession, RecallEventType, InputType
from coherent.core.memory.experience_manager import ExperienceManager

# Visualization Utilities
from viz_utils import complex_to_hsv_rgb

# --- MOCKS (For Standalone Run) ---
class StreamlitMockParser:
    def parse_to_vector(self, text: str) -> np.ndarray:
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

# --- CONFIG & STATE ---
st.set_page_config(page_title="COHERENT Simulator v2.0", layout="wide")
st.title("COHERENT: MemorySpace Recall Simulator v2.0")

if 'simulator' not in st.session_state:
    exp_mgr = StreamlitMockExperienceManager()
    parser = StreamlitMockParser()
    sim = RecallFirstSimulator(exp_mgr, parser=parser)
    # Bootstrap Static Memory
    vec_math = np.zeros(64); vec_math[0]=1.0; vec_math[1]=0.5; vec_math[2]=0.5
    vec_math = vec_math / np.linalg.norm(vec_math)
    sim.layer2_static.add(vec_math, {"id": "concept_algebra_basic"})
    st.session_state.simulator = sim

if 'session_data' not in st.session_state:
    st.session_state.session_data = None # RecallSession
if 'current_event_idx' not in st.session_state:
    st.session_state.current_event_idx = 0

# --- SIDEBAR: CONTROLS ---
st.sidebar.header("1. Execution Control")

SCENARIOS = {
    "Manual": "",
    "E1: Direct Instruction": "3x + 5x を計算せよ",
    "E2: Paraphrase": "3x と 5x をまとめて",
    "E3: Causal Instruction": "まず両辺から3を引き、その後2で割れ",
    "E4: Ambiguity Refusal": "いい感じに解いて",
    "E7: Experience Reuse": "7x + x をまとめよ"
}

selected_scenario = st.sidebar.selectbox("Select Scenario", list(SCENARIOS.keys()))
init_val = SCENARIOS[selected_scenario] if selected_scenario != "Manual" else ""
task_input = st.sidebar.text_area("Instruction", value=init_val)

uploaded_file = st.sidebar.file_uploader("Attach Context/File", type=['txt', 'md'])
file_content = ""
if uploaded_file is not None:
    file_content = uploaded_file.read().decode("utf-8")
    st.sidebar.info(f"Loaded {uploaded_file.name}")

if st.sidebar.button("Execute Task"):
    with st.spinner("Running Recall Pipeline..."):
        # Reset Simulator for cleanliness (optional, but good for demo)
        st.session_state.simulator = RecallFirstSimulator(StreamlitMockExperienceManager(), StreamlitMockParser())
        # Re-Bootstrap
        vec_math = np.zeros(64); vec_math[0]=1.0; vec_math[1]=0.5; vec_math[2]=0.5
        vec_math = vec_math / np.linalg.norm(vec_math)
        st.session_state.simulator.layer2_static.add(vec_math, {"id": "concept_algebra_basic"})
        
        # Run
        final_input = task_input
        input_type = InputType.TEXT
        
        if uploaded_file:
            # If file is present, we treat it as the content or append it.
            # Simplified: if text empty, use file. If both, append file to text.
            if not task_input.strip():
                final_input = file_content
                input_type = InputType.FILE
            else:
                final_input = f"{task_input}\n\n{file_content}"
                input_type = InputType.FILE # Mixed content treated as file/complex
        
        session = st.session_state.simulator.start_session(final_input, input_type=input_type)
        try:
            st.session_state.simulator.execute_pipeline()
            st.session_state.session_data = session
            st.session_state.current_event_idx = 0
            st.success("Execution Complete")
        except Exception as e:
            st.error(f"Execution Failed: {e}")

# --- RESULT SECTION ---
if st.session_state.session_data:
    st.divider()
    res_col1, res_col2 = st.columns([3, 1])
    
    with res_col1:
        st.subheader("Execution Result")
        result_text = st.session_state.session_data.execution_result
        if result_text:
            st.success(f"{result_text}")
        else:
            st.info("No result produced (Review or Suppression).")
            
    with res_col2:
        st.subheader("Inference Source")
        source = st.session_state.session_data.inference_source
        if source == "Holographic (Recall)":
            st.markdown("### 🔮 Recall")
            st.caption("Retrieved from Memory")
        elif source == "Logic (Computation)":
            st.markdown("### ⚙️ Logic")
            st.caption("Computed by Engine")
        else:
            st.markdown("### ❓ Unknown")

# --- MAIN: A-AXIS (TIMELINE) ---
st.header("A-Axis: MemorySpace Layers (Causal Structure)")

if st.session_state.session_data and st.session_state.session_data.events:
    events = st.session_state.session_data.events
    
    # 1. Prepare Data for Altair
    # Map layers to Y-axis order
    layer_order = ["MS-L3", "MS-L2", "MS-L1"]
    
    timeline_data = []
    for i, evt in enumerate(events):
        timeline_data.append({
            "step": i,
            "layer": evt.memory_layer,
            "type": evt.event_type.value,
            "timestamp": evt.timestamp,
            "selected": (i == st.session_state.current_event_idx)
        })
    
    df_timeline = pd.DataFrame(timeline_data)
    
    # 2. Render Interactive Chart
    # We use a selection interval or just click-to-select logic if possible.
    # Streamlit doesn't support bi-directional click back from Altair easily without custom component or hacks.
    # We will use a SLIDER below to control 'current_idx', and chart just visualizes.
    
    base = alt.Chart(df_timeline).encode(
        x=alt.X('step:O', title="Step ID"),
        y=alt.Y('layer:N', sort=layer_order, title="Memory Layer"),
        tooltip=['step', 'type', 'layer']
    )
    
    points = base.mark_circle(size=200).encode(
        color=alt.condition(
            alt.datum.selected, 
            alt.value('red'),  # Selected
            alt.value('steelblue') # Default
        )
    )
    
    text = base.mark_text(dy=-15, size=10).encode(text='type')
    
    chart = (points + text).properties(height=200, width='container')
    st.altair_chart(chart, use_container_width=True)
    
    # 3. Control Slider
    max_idx = len(events) - 1
    selected_idx = st.slider("Select Step to Inspect", 0, max_idx, st.session_state.current_event_idx)
    st.session_state.current_event_idx = selected_idx
    
    current_event = events[selected_idx]
    
else:
    st.info("Execute a task to generate the A-Axis Timeline.")
    current_event = None

# --- MAIN: B-AXIS (HOLOGRAPHIC MEMORY) ---
st.divider()
st.header("B-Axis: Holographic Representation (Physical Basis)")

if current_event:
    col_vis, col_meta = st.columns([2, 1])
    
    with col_vis:
        st.subheader(f"HM-1: Semantic Field @ {current_event.memory_layer}")
        
        # Visualize Field (Synthesized based on event type)
        # We assume a Virtual Field for visualization
        size = 128
        field = np.zeros((size, size), dtype=np.complex64)
        x = np.linspace(-10, 10, size)
        y = np.linspace(-10, 10, size)
        X, Y = np.meshgrid(x, y)
        
        etype = current_event.event_type.value
        metrics = current_event.metrics
        
        # Logic from previous app.py but refined
        if "INPUT" in etype or "QUERY" in etype:
            # Wave packet
            R = np.sqrt(X**2 + Y**2)
            phase = (current_event.timestamp % 1.0) * 2 * np.pi
            field += 0.8 * np.exp(-0.5 * (R**2)) * np.exp(1j * (phase - R))
            
        elif "RESONANCE" in etype or "FILTER" in etype:
            # Interference Pattern
            res = metrics.get('total_resonance', 0.5)
            # Source 1 (Query)
            R1 = np.sqrt(X**2 + Y**2)
            field += 0.5 * np.exp(-0.5 * R1**2)
            # Source 2 (Memory)
            R2 = np.sqrt((X-3)**2 + (Y-2)**2)
            field += res * np.exp(-0.5 * R2**2) * np.exp(1j * np.pi/4)
            
        elif "DECISION" in etype or "ACTION" in etype:
            # Stable State
            R = np.sqrt(X**2 + Y**2)
            field += 1.0 * np.exp(-0.2 * R**2) # Broad stable field
            
        # Draw
        rgb = complex_to_hsv_rgb(field)
        st.image(rgb, caption="Synthesized Holographic Field", use_container_width=True)
        
    with col_meta:
        st.subheader("HM-2 & HM-3: State Metrics")
        
        # Metrics Display
        st.json(current_event.metrics)
        
        # HM-2: Resonance Spectrum Bar Chart
        if "dhm_top_k" in metrics or "shm_top_k" in metrics:
            st.markdown("#### Resonance Spectrum")
            
            data = []
            if "dhm_top_k" in metrics:
                for item in metrics["dhm_top_k"]:
                    data.append({"Source": "DHM", "ID": item['id'], "Score": item['score']})
            if "shm_top_k" in metrics:
                for item in metrics["shm_top_k"]:
                    data.append({"Source": "SHM", "ID": item['id'], "Score": item['score']})
            
            if data:
                df_res = pd.DataFrame(data)
                chart_res = alt.Chart(df_res).mark_bar().encode(
                    x='Score:Q',
                    y=alt.Y('ID:N', sort='-x'),
                    color='Source:N',
                    tooltip=['Source', 'ID', 'Score']
                )
                st.altair_chart(chart_res, use_container_width=True)
            else:
                st.write("No strong resonance detected.")
                
        # HM-3: Decision Metrics
        if "ambiguity" in metrics:
             st.markdown("#### Decision Trace")
             col1, col2, col3 = st.columns(3)
             col1.metric("Resonance", f"{metrics.get('resonance',0):.2f}")
             col2.metric("Ambiguity", f"{metrics.get('ambiguity',0):.2f}")
             col3.metric("Margin", f"{metrics.get('margin',0):.2f}")

else:
    st.write("Waiting for execution...")

