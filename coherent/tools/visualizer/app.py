import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import time
from coherent.tools.visualizer.models import TraceEvent
from coherent.tools.visualizer.state_manager import StateManager
from coherent.tools.visualizer.simulator import TraceSimulator

st.set_page_config(page_title="Holographic Memory Visualizer", layout="wide")

st.title("🧠 Real-Time Holographic Memory Visualizer")

# Session State Init
if "state_manager" not in st.session_state:
    st.session_state.state_manager = StateManager()

if "simulator" not in st.session_state:
    st.session_state.simulator = TraceSimulator()
    
if "auto_play" not in st.session_state:
    st.session_state.auto_play = False

# Sidebar Controls
st.sidebar.header("Controls")
if st.sidebar.button("Step Forward"):
    try:
        # Generate one sequence of events (one step worth)
        events = list(st.session_state.simulator.generate_step_sequence())
        for e in events:
            st.session_state.state_manager.process_event(e)
            st.session_state.last_event = e
    except StopIteration:
        st.warning("Simulation Ended")

auto_play = st.sidebar.checkbox("Auto Play", value=False)

# Main Dashboard
col_2d, col_3d = st.columns(2)

state = st.session_state.state_manager.state

with col_2d:
    st.subheader("2D Holographic Signal (Interference)")
    
    # Render Heatmap
    # Amplitude
    fig_amp = px.imshow(
        state.amplitude_map, 
        color_continuous_scale="Viridis",
        title="Amplitude Map"
    )
    st.plotly_chart(fig_amp, use_container_width=True)
    
    # Phase
    fig_phase = px.imshow(
        state.phase_map, 
        color_continuous_scale="Twilight",
        title="Phase Map"
    )
    st.plotly_chart(fig_phase, use_container_width=True)

with col_3d:
    st.subheader("3D Memory Space Topology")
    
    # Prepare Data
    ids = list(state.memory_coords.keys())
    if ids:
        x = [state.memory_coords[i][0] for i in ids]
        y = [state.memory_coords[i][1] for i in ids]
        z = [state.memory_activation.get(i, 0.0) for i in ids]
        
        # Color by activation
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers+text',
            text=ids,
            marker=dict(
                size=8,
                color=z,
                colorscale='Plasma',
                opacity=0.8
            )
        )])
        fig_3d.update_layout(scene=dict(
            xaxis_title='Memory X',
            yaxis_title='Memory Y',
            zaxis_title='Resonance (Z)'
        ))
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.info("No memory traces yet.")

# Event Log
with st.expander("Event Log"):
    if "last_event" in st.session_state:
        st.json(st.session_state.last_event.model_dump())

# Auto-play Logic (Streamlit rerun trick)
if auto_play:
    time.sleep(0.5)
    events = list(st.session_state.simulator.generate_step_sequence())
    for e in events:
        st.session_state.state_manager.process_event(e)
        st.session_state.last_event = e
    st.rerun()
