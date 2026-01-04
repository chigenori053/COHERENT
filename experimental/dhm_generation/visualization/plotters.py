"""
DHM Visualization Plotters

Implements:
- Trajectory Plot (Spec 4.1)
- Success State Geometry Plot (Spec 4.2)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Dict, Any, Tuple

class VisualizationPlotter:
    def __init__(self, projector):
        self.projector = projector

    def plot_trajectory(
        self, 
        trajectory: np.ndarray, 
        target_label: str, 
        ax=None
    ):
        """
        Renders H_T -> ... -> H_0 trajectory.
        trajectory: (steps, D) complex vectors.
                    trajectory[0] is noisy start (T)
                    trajectory[-1] is final (0)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            
        # Project
        coords = self.projector.transform(trajectory)
        
        # Plot path
        # Color gradient from Start (Light) to End (Dark)
        num_points = coords.shape[0]
        
        # We want to emphasize direction
        # Plot segments
        for i in range(num_points - 1):
            # i=0 is start (noisy), i=End is final
            # Let's interactively color
            prog = i / (num_points - 1)
            color = plt.cm.viridis(prog) # Start purple, end yellow? Or reversed
            # Using 'cool' colormap: Cyan (start) -> Magenta (end)
            color = plt.cm.cool(prog)
            
            ax.plot(
                coords[i:i+2, 0], coords[i:i+2, 1], 
                color=color, alpha=0.7, linewidth=2
            )
            
        # Markers
        ax.scatter(coords[0, 0], coords[0, 1], c='red', marker='x', label='Start (Noisy)', zorder=5)
        ax.scatter(coords[-1, 0], coords[-1, 1], c='green', marker='o', label='End (Refined)', zorder=5)
        
        ax.set_title(f"Refinement Trajectory: Target {target_label}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def plot_geometry(
        self, 
        final_state: np.ndarray, 
        oracle_matrix: np.ndarray, 
        oracle_labels: List[str],
        target_label: str, 
        predicted_label: str,
        ax=None
    ):
        """
        Renders final state vs all oracle symbols.
        final_state: (1, D) or (D,)
        oracle_matrix: (26, D)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            
        if final_state.ndim == 1:
            final_state = final_state.reshape(1, -1)
            
        # Project everything together
        # Note: Projector must handle the combined batch or transform separately
        # We assume projector handles (N, D).
        
        oracle_coords = self.projector.transform(oracle_matrix)
        final_coord = self.projector.transform(final_state)[0]
        
        # Plot Oracles (Reference)
        ax.scatter(oracle_coords[:, 0], oracle_coords[:, 1], c='gray', alpha=0.5, label='Candidates')
        
        # Annotate Oracles
        for idx, label in enumerate(oracle_labels):
            is_target = (label == target_label)
            weight = 'bold' if is_target else 'normal'
            fontsize = 12 if is_target else 9
            color = 'black'
            
            if is_target:
                color = 'blue'
            
            ax.text(
                oracle_coords[idx, 0], oracle_coords[idx, 1], 
                label, fontsize=fontsize, weight=weight, color=color
            )
            
        # Plot Final State
        ax.scatter(final_coord[0], final_coord[1], c='red', marker='*', s=200, label=f'Generated ({predicted_label})', zorder=10)
        
        # Draw line to predicted oracle if different or same
        # Find index of predicted
        if isinstance(oracle_labels, np.ndarray):
             oracle_labels = oracle_labels.tolist()
             
        if predicted_label in oracle_labels:
            pred_idx = oracle_labels.index(predicted_label)
            pred_coord = oracle_coords[pred_idx]
            ax.plot([final_coord[0], pred_coord[0]], [final_coord[1], pred_coord[1]], 'r--', alpha=0.5)

        title = f"State Geometry: Target {target_label} -> Pred {predicted_label}"
        if target_label != predicted_label:
            title += " (MISMATCH)"
            
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.3)
        ax.legend()
        return ax
