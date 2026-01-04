"""
Visualization App

Generates plots from experiment logs strictly according to Spec.
Visualizes:
1. Trajectory (H_T ... -> H_0)
2. Geometry (Final State Context)
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

try:
    from .plotter_core import DHMStateProjector
    from .plotters import VisualizationPlotter
except ImportError:
    # Allow running as script from root
    from experimental.dhm_generation.visualization.plotter_core import DHMStateProjector
    from experimental.dhm_generation.visualization.plotters import VisualizationPlotter

def load_latest_vectors(log_dir):
    """Load all .npz files from the latest experiment session."""
    vector_dir = os.path.join(log_dir, "vectors")
    if not os.path.exists(vector_dir):
        print(f"No vector logs found in {vector_dir}")
        return []

    # Simple heuristic: load all npz in the folder.
    # Ideally should group by session timestamp.
    # Assuming user clears or we just process all.
    # Let's process ALL .npz files found.
    files = glob.glob(os.path.join(vector_dir, "*.npz"))
    if not files:
        print("No .npz files found.")
        return []
        
    print(f"Found {len(files)} vector log files.")
    
    # Sort files by modification time (newest first) to establish target dimension
    files.sort(key=os.path.getmtime, reverse=True)
    
    data_list = []
    target_dim = None
    
    for f in files:
        try:
            data = np.load(f, allow_pickle=True)
            # Check dimension from trajectory shape (Time, Dim)
            dim = data['trajectory'].shape[1]
            
            if target_dim is None:
                target_dim = dim
                print(f"Targeting dimension: {target_dim} (from newest file)")
            
            if dim == target_dim:
                data_list.append(data)
            else:
                # Skip files with mismatching dimension
                continue
                
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    print(f"Loaded {len(data_list)} logs with dimension {target_dim}.")
    return data_list

def generate_visualizations(log_dir="experimental/dhm_generation/logs"):
    plot_dir = os.path.join(log_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    print("Loading logs...")
    runs = load_latest_vectors(log_dir)
    if not runs:
        return

    print("Fitting Global PCA Projector...")
    # Collect all states from all runs to build a common basis (Spec 3.2)
    # Includes: trajectory states + oracle states
    all_states = []
    for run in runs:
        all_states.append(run['trajectory']) # (T+1, D)
        # oracle matrix is same for all, strictly speaking, but let's include once
    
    # Also add oracle matrix from first run
    if runs:
        all_states.append(runs[0]['oracle_matrix'])
        
    # Concatenate
    training_data = np.concatenate(all_states, axis=0)
    
    projector = DHMStateProjector(output_dim=2)
    try:
        projector.fit(training_data)
    except ImportError:
        print("Scikit-learn not installed. Cannot perform PCA.")
        return

    print(f"PCA Variance Explained: {projector.get_explained_variance()}")
    
    plotter = VisualizationPlotter(projector)

    print("Generating Plots...")
    for run in runs:
        target = str(run['target_symbol'])
        pred = str(run['final_symbol'])
        
        # 1. Trajectory Plot (Spec 4.1)
        fig, ax = plt.subplots(figsize=(8,8))
        plotter.plot_trajectory(run['trajectory'], target, ax=ax)
        fig.savefig(os.path.join(plot_dir, f"traj_{target}.png"))
        plt.close(fig)
        
        # 2. Geometry Plot (Spec 4.2)
        fig, ax = plt.subplots(figsize=(10,8))
        plotter.plot_geometry(
            run['trajectory'][-1], # Final state
            run['oracle_matrix'],
            run['oracle_labels'],
            target,
            pred,
            ax=ax
        )
        fig.savefig(os.path.join(plot_dir, f"geo_{target}.png"))
        plt.close(fig)
        
    print(f"Plots saved to {plot_dir}")

if __name__ == "__main__":
    generate_visualizations()
