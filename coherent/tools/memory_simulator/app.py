import streamlit as st
import numpy as np
import time
import pandas as pd
import altair as alt # Use altair for simple charts if needed, or st.line_chart

from model import MemorySpaceSystem
from scenarios import setup_scenario_a, setup_scenario_b, setup_scenario_c
from viz_utils import complex_to_hsv_rgb, compute_phase_variance

# Page Config
st.set_page_config(page_title="Holographic Memory Simulator", layout="wide")

st.title("Holographic Memory / MemorySpace Simulator v1.0")

# --- Sidebar Controls ---
st.sidebar.header("Configuration")

scenario_choice = st.sidebar.selectbox(
    "Select Scenario",
    ["Manual", "Scenario A: Resonance", "Scenario B: SHM Block/Filter", "Scenario C: CHM Violation"]
)

# Resolution
res = st.sidebar.select_slider("Resolution", options=[64, 128, 256], value=128)

# Visualization Params
st.sidebar.subheader("Visualization")
gamma = st.sidebar.slider("Gamma (Brightness)", 0.1, 1.0, 0.6)
show_stability = st.sidebar.checkbox("Show Stability (Saturation)", value=True)

# Simulation Control
st.sidebar.subheader("Simulation")
running = st.sidebar.checkbox("Run Simulation", value=False)
speed = st.sidebar.slider("Speed (dt)", 0.01, 0.5, 0.1)

# Causal Control (Interactive)
user_gate = st.sidebar.slider("CHM Gate (Interactive)", 0.0, 1.0, 1.0)


# --- System Initialization ---
if 'system' not in st.session_state:
    st.session_state.system = MemorySpaceSystem(size=res)
    st.session_state.initialized_scenario = None

sys = st.session_state.system

# Resolution change check
if sys.size != res:
    sys = MemorySpaceSystem(size=res)
    st.session_state.system = sys
    st.session_state.initialized_scenario = None

# Scenario Setup Logic
if scenario_choice != st.session_state.initialized_scenario:
    if scenario_choice == "Scenario A: Resonance":
        msg = setup_scenario_a(sys)
        st.success(msg)
    elif scenario_choice == "Scenario B: SHM Block/Filter":
        msg = setup_scenario_b(sys)
        st.success(msg)
    elif scenario_choice == "Scenario C: CHM Violation":
        msg = setup_scenario_c(sys)
        sys.chm.gate_state = 0.0 # Force start closed
        st.success(msg)
    elif scenario_choice == "Manual":
        sys.reset()
        st.info("Manual Mode: Add DHMs via code or assume empty.")
    
    st.session_state.initialized_scenario = scenario_choice

# Interactive Gate Override for Manual/C
if scenario_choice == "Scenario C: CHM Violation" or scenario_choice == "Manual":
    sys.chm.active = True
    sys.chm.gate_state = user_gate
    st.sidebar.caption(f"Gate State: {sys.chm.gate_state}")

# --- Main Layout ---
col1, col2 = st.columns([2, 1])

# Container for the image
with col1:
    hologram_placeholder = st.empty()

with col2:
    st.subheader("Metrics")
    metric_placeholder = st.empty()
    st.subheader("Resonance History")
    chart_placeholder = st.empty()

# --- Simulation Loop ---
if running:
    # Run a burst of steps or just one per frame? Streamlit reruns script on interaction.
    # To animate, we use a loop with st.empty()
    
    while running:
        # Step
        sys.step(dt=speed)
        
        # Calc Stability Map if requested
        stability_map = None
        if show_stability:
            stability_map = compute_phase_variance(sys.field)
        
        # Render
        # Normalize amp by finding max in recent history or current?
        # Use a fixed reasonable max or dynamic
        max_amp = np.max(np.abs(sys.field)) 
        if max_amp < 1e-6: max_amp = 1.0
        
        rgb_img = complex_to_hsv_rgb(
            sys.field, 
            max_amp=max_amp, 
            gamma=gamma, 
            stability_map=stability_map
        )
        
        # Update UI
        hologram_placeholder.image(rgb_img, caption=f"Time: {sys.time:.2f}", use_container_width=True)
        
        # Metrics
        total_energy = sys.resonance_history[-1] if sys.resonance_history else 0
        metric_placeholder.markdown(f"""
        **Time**: {sys.time:.2f}  
        **Total Resonance**: {total_energy:.4f}  
        **Max Amplitude**: {max_amp:.4f}
        """)
        
        # Chart
        if len(sys.resonance_history) > 0:
            hist_df = pd.DataFrame({
                "Step": range(len(sys.resonance_history)),
                "Resonance": sys.resonance_history
            })
            # Limit history
            if len(hist_df) > 100:
                hist_df = hist_df.iloc[-100:]
            
            c = alt.Chart(hist_df).mark_line().encode(
                x='Step',
                y='Resonance'
            ).properties(height=200)
            chart_placeholder.altair_chart(c, use_container_width=True)
            
        time.sleep(0.05) # Cap FPS slightly
        
        # Streamlit handles breaking the loop if user stops via UI?
        # Actually standard Streamlit halts on interaction. 
        # But 'while running' inside the script might block interaction unless we use st.experimental_rerun?
        # Or simple 'st.empty()' loop works until interaction triggers rerun.
        # We will assume standard loop pattern.
else:
    # Static render of current state
    max_amp = np.max(np.abs(sys.field)) if len(sys.field) > 0 else 1.0
    if max_amp < 1e-6: max_amp = 1.0
    
    stability_map = compute_phase_variance(sys.field) if show_stability else None
    rgb_img = complex_to_hsv_rgb(sys.field, max_amp, gamma, stability_map)
    hologram_placeholder.image(rgb_img, caption="Simulation Paused", use_container_width=True)

