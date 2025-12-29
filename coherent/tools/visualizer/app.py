import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import time
import tempfile
import cv2
import os

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

if "recording" not in st.session_state:
    st.session_state.recording = False
    st.session_state.recorded_frames = []

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
st.sidebar.markdown("---")
st.sidebar.header("Recording")

def toggle_recording():
    st.session_state.recording = not st.session_state.recording
    if st.session_state.recording:
        st.session_state.recorded_frames = []
        st.toast("Recording Started 🔴")
    else:
        st.toast("Recording Stopped ⏹️")

record_btn_label = "Stop Recording ⏹️" if st.session_state.recording else "Start Recording 🔴"
st.sidebar.button(record_btn_label, on_click=toggle_recording)

if st.session_state.recorded_frames:
    st.sidebar.info(f"{len(st.session_state.recorded_frames)} frames recorded.")
    
    if not st.session_state.recording:
        if st.sidebar.button("Export Video 🎬"):
            with st.spinner("Generating Video..."):
                frames = st.session_state.recorded_frames
                if not frames:
                    st.error("No frames to export.")
                else:
                    # Create temp file
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    filename = tfile.name
                    tfile.close()

                    # Setup Video Writer
                    # We need to know image size. Let's write the first frame to check.
                    # Note: We are storing 'fig' objects.
                    
                    # Helper to get image bytes
                    try:
                        import kaleido
                    except ImportError:
                         st.error("Kaleido not found. Please install kaleido.")
                         st.stop()
                        
                    # Write first frame to determine dimensions
                    # We use a static width/height for video consistency
                    width, height = 800, 600
                    
                    # Define codec
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                    out = cv2.VideoWriter(filename, fourcc, 2.0, (width, height)) # 2 FPS

                    for i, fig in enumerate(frames):
                        # Convert Plotly fig to image (png) bytes
                        img_bytes = fig.to_image(format="png", width=width, height=height, engine="kaleido")
                        
                        # Convert bytes to numpy array for opencv
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        out.write(img)
                    
                    out.release()
                    
                    # Read back for download
                    with open(filename, "rb") as f:
                        video_bytes = f.read()
                        
                    st.sidebar.download_button(
                        label="Download Video 📥",
                        data=video_bytes,
                        file_name="memory_trace_video.mp4",
                        mime="video/mp4"
                    )
                    
                    # Cleanup
                    os.unlink(filename)

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
        
        # Color by phase
        phase_map = {
            "Phase 1": "#1f77b4", # Blue
            "Phase 2": "#2ca02c", # Green
            "Phase 3": "#d62728", # Red
            "Unknown": "#7f7f7f"  # Gray
        }
        
        phases = [state.memory_phase.get(i, "Unknown") for i in ids]
        colors = [phase_map.get(p, "#7f7f7f") for p in phases]
        
        fig_3d = go.Figure()
        
        # Split data by unique phase for legend
        unique_phases = sorted(list(set(phases)))
        
        for p_key in unique_phases:
            # Filter indices for this phase
            indices = [i for i, p in enumerate(phases) if p == p_key]
            
            p_x = [x[i] for i in indices]
            p_y = [y[i] for i in indices]
            p_z = [z[i] for i in indices]
            p_ids = [ids[i] for i in indices]
            
            fig_3d.add_trace(go.Scatter3d(
                x=p_x, y=p_y, z=p_z,
                mode='markers',
                name=p_key, # Legend Label
                hovertext=[f"ID: {mid}<br>Phase: {p_key}" for mid in p_ids],
                hoverinfo="text",
                marker=dict(
                    size=8,
                    color=phase_map.get(p_key, "#7f7f7f"),
                    opacity=0.8
                )
            ))
        fig_3d.update_layout(
            scene=dict(
                xaxis_title='Memory X',
                yaxis_title='Memory Y',
                zaxis_title='Resonance (Z)'
            ),
            uirevision='constant' # Persist camera angle
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Capture frame if recording
        if st.session_state.recording:
            # We copy the figure to ensure we capture independent states
            st.session_state.recorded_frames.append(go.Figure(fig_3d))
            
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
