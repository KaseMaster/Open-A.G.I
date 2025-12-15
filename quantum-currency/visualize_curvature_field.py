#!/usr/bin/env python3
"""
Visualization of Resonant Curvature Field
Implements curvature heatmap layer for Global Resonance Dashboard
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.field.curvature.cci_core import CurvatureCoherenceIntegrator
from src.field.curvature.q_projection import QProjection

def generate_sample_curvature_field(grid_size=50):
    """Generate a sample curvature field for visualization"""
    # Create coordinate grids
    x = np.linspace(-5, 5, grid_size)
    y = np.linspace(-5, 5, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Generate sample curvature data with multiple peaks
    # This simulates the R_Ω tensor magnitude at different lattice points
    Z = (np.exp(-(X**2 + Y**2)/4) * np.cos(X) * np.sin(Y) + 
         0.5 * np.exp(-((X-2)**2 + (Y-1)**2)/2) +
         0.3 * np.exp(-((X+1)**2 + (Y+2)**2)/3))
    
    # Scale to match our curvature magnitudes
    Z = Z * 2.16e-62
    
    return X, Y, Z

def create_curvature_heatmap(X, Y, Z):
    """Create a heatmap visualization of the curvature field"""
    # Create custom colormap for better visualization
    colors = ['navy', 'blue', 'cyan', 'yellow', 'orange', 'red']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('curvature', colors, N=n_bins)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    im = ax.imshow(Z, extent=[X.min(), X.max(), Y.min(), Y.max()], 
                   cmap=cmap, origin='lower', aspect='equal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Curvature Magnitude R_Ω', rotation=270, labelpad=20)
    
    # Add contour lines
    contours = ax.contour(X, Y, Z, levels=8, colors='white', alpha=0.4, linewidths=0.5)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%.2e')
    
    # Labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Resonant Curvature Field Visualization\nR_Ω Tensor Magnitude Distribution')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    return fig, ax

def add_quantum_state_annotations(ax, num_states=5):
    """Add quantum state annotations to the visualization"""
    # Sample quantum states for annotation
    states = [
        {'n': 2, 'l': 1, 'm': 0, 's': 0.5, 'x': -2, 'y': 1},
        {'n': 3, 'l': 2, 'm': 1, 's': -0.5, 'x': 3, 'y': -2},
        {'n': 1, 'l': 0, 'm': 0, 's': 0.5, 'x': 0, 'y': 3},
        {'n': 4, 'l': 3, 'm': -2, 's': -0.5, 'x': -3, 'y': -1},
        {'n': 2, 'l': 1, 'm': 1, 's': 0.5, 'x': 2, 'y': 2},
    ]
    
    # Add annotations
    for i, state in enumerate(states[:num_states]):
        ax.annotate(f'Q({state["n"]},{state["l"]},{state["m"]},{state["s"]})', 
                   xy=(state['x'], state['y']), 
                   xytext=(state['x'] + 0.5, state['y'] + 0.5),
                   arrowprops=dict(arrowstyle='->', color='white', alpha=0.7),
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7),
                   fontsize=8, color='white')

def create_3d_curvature_surface(X, Y, Z):
    """Create a 3D surface plot of the curvature field"""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create surface plot
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    
    # Add contour projections
    ax.contour(X, Y, Z, zdir='z', offset=Z.min(), cmap='viridis', alpha=0.5)
    ax.contour(X, Y, Z, zdir='x', offset=X.min(), cmap='viridis', alpha=0.5)
    ax.contour(X, Y, Z, zdir='y', offset=Y.max(), cmap='viridis', alpha=0.5)
    
    # Labels
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Curvature Magnitude R_Ω')
    ax.set_title('3D Resonant Curvature Field')
    
    # Add colorbar
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    return fig, ax

def main():
    """Main function to generate curvature visualizations"""
    print("🔬 Generating Resonant Curvature Field Visualizations...")
    print("=" * 55)
    
    # Generate sample curvature field
    print("Generating sample curvature field...")
    X, Y, Z = generate_sample_curvature_field()
    
    # Create 2D heatmap
    print("Creating 2D curvature heatmap...")
    fig1, ax1 = create_curvature_heatmap(X, Y, Z)
    add_quantum_state_annotations(ax1)
    
    # Save the plot
    heatmap_path = 'curvature_heatmap.png'
    fig1.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"✅ 2D heatmap saved to: {heatmap_path}")
    
    # Create 3D surface plot
    print("Creating 3D curvature surface...")
    try:
        fig2, ax2 = create_3d_curvature_surface(X, Y, Z)
        surface_path = 'curvature_surface.png'
        fig2.savefig(surface_path, dpi=300, bbox_inches='tight')
        print(f"✅ 3D surface saved to: {surface_path}")
    except ImportError:
        print("⚠️  3D plotting not available (mpl_toolkits.mplot3d not found)")
    
    # Generate sample data for dashboard integration
    print("Generating sample data for dashboard integration...")
    sample_data = {
        "visualization_type": "curvature_field",
        "data_points": int(Z.size),
        "min_curvature": float(Z.min()),
        "max_curvature": float(Z.max()),
        "mean_curvature": float(Z.mean()),
        "timestamp": "2025-11-09T15:30:00Z",
        "units": "[L]⁻²",
        "sample_coordinates": [
            {"x": float(X[0, 0]), "y": float(Y[0, 0]), "curvature": float(Z[0, 0])},
            {"x": float(X[-1, -1]), "y": float(Y[-1, -1]), "curvature": float(Z[-1, -1])},
            {"x": float(X[Z.shape[0]//2, Z.shape[1]//2]), 
             "y": float(Y[Z.shape[0]//2, Z.shape[1]//2]), 
             "curvature": float(Z[Z.shape[0]//2, Z.shape[1]//2])}
        ]
    }
    
    # Save sample data
    data_path = 'curvature_dashboard_data.json'
    with open(data_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    print(f"✅ Dashboard data saved to: {data_path}")
    
    print("\n🎨 Visualization Summary:")
    print(f"   Grid size: {Z.shape[0]} × {Z.shape[1]}")
    print(f"   Curvature range: {Z.min():.2e} to {Z.max():.2e}")
    print(f"   Mean curvature: {Z.mean():.2e}")
    print(f"   Data points: {Z.size:,}")
    
    print("\n🚀 Curvature visualization generation complete!")
    print("   Next steps:")
    print("   1. Integrate with Global Resonance Dashboard")
    print("   2. Enable real-time curvature resonance monitoring")
    print("   3. Add interactive controls for quantum state selection")

if __name__ == "__main__":
    main()