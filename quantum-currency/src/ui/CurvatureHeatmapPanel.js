import React, { useState, useEffect } from 'react';

const CurvatureHeatmapPanel = () => {
  const [metrics, setMetrics] = useState(null);
  const [latticeData, setLatticeData] = useState([]);
  const [viewMode, setViewMode] = useState('2D');
  const [isLoading, setIsLoading] = useState(true);

  // Simulate WebSocket data for demonstration
  useEffect(() => {
    const interval = setInterval(() => {
      // Generate mock metrics
      const mockMetrics = {
        timestamp: Date.now(),
        rsi: 0.85 + Math.random() * 0.14,
        cs: 0.90 + Math.random() * 0.09,
        gas: 0.95 + Math.random() * 0.04,
        R_Omega_magnitude: 2.15e-62 + Math.random() * 0.1e-62,
        T_Omega: 7.23e-38 + Math.random() * 0.5e-38,
        safe_mode: false,
        stability_state: 'STABLE'
      };
      
      setMetrics(mockMetrics);
      updateLatticeData(mockMetrics);
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  // Simulate lattice data based on curvature metrics
  const updateLatticeData = (metrics) => {
    // Generate sample lattice nodes
    const nodes = [];
    const gridSize = 20;
    
    for (let x = 0; x < gridSize; x++) {
      for (let y = 0; y < gridSize; y++) {
        // Calculate curvature based on distance from center and metrics
        const centerX = gridSize / 2;
        const centerY = gridSize / 2;
        const distance = Math.sqrt(Math.pow(x - centerX, 2) + Math.pow(y - centerY, 2));
        
        // Base curvature influenced by R_Ω magnitude
        const baseCurvature = metrics.R_Omega_magnitude * (1 - distance / gridSize);
        
        // Determine if this is a nodal surface point
        const isNodalSurface = distance % 3 < 0.5;
        
        nodes.push({
          x,
          y,
          curvature: baseCurvature,
          nodal_surface: isNodalSurface
        });
      }
    }
    
    setLatticeData(nodes);
    setIsLoading(false);
  };

  // Render 2D heatmap
  const render2DHeatmap = () => {
    if (isLoading || !latticeData.length) {
      return <div className="heatmap-placeholder">Loading curvature data...</div>;
    }

    // Find min/max curvature for color scaling
    const curvatures = latticeData.map(node => node.curvature);
    const minCurvature = Math.min(...curvatures);
    const maxCurvature = Math.max(...curvatures);
    const range = maxCurvature - minCurvature || 1;

    return (
      <div className="heatmap-2d">
        {latticeData.map((node, index) => {
          // Calculate color based on curvature value
          const normalized = (node.curvature - minCurvature) / range;
          const hue = 240 - (normalized * 120); // Blue to red
          const backgroundColor = node.nodal_surface 
            ? `hsl(${hue}, 100%, 30%)` 
            : `hsl(${hue}, 70%, 50%)`;
          
          return (
            <div
              key={index}
              className={`heatmap-cell ${node.nodal_surface ? 'nodal-surface' : ''}`}
              style={{
                backgroundColor,
                gridColumn: node.x + 1,
                gridRow: node.y + 1
              }}
              title={`(${node.x}, ${node.y}) - Curvature: ${node.curvature.toExponential(2)}`}
            />
          );
        })}
      </div>
    );
  };

  // Render 3D surface (simplified representation)
  const render3DSurface = () => {
    if (isLoading || !latticeData.length) {
      return <div className="surface-placeholder">Loading 3D surface...</div>;
    }

    // Find min/max curvature for height scaling
    const curvatures = latticeData.map(node => node.curvature);
    const minCurvature = Math.min(...curvatures);
    const maxCurvature = Math.max(...curvatures);
    const range = maxCurvature - minCurvature || 1;

    return (
      <div className="surface-3d">
        {latticeData.map((node, index) => {
          // Calculate height based on curvature
          const normalized = (node.curvature - minCurvature) / range;
          const height = normalized * 100; // Max height 100px
          
          // Calculate color
          const hue = 240 - (normalized * 120);
          const backgroundColor = node.nodal_surface 
            ? `hsl(${hue}, 100%, 30%)` 
            : `hsl(${hue}, 70%, 50%)`;
          
          return (
            <div
              key={index}
              className={`surface-cell ${node.nodal_surface ? 'nodal-surface' : ''}`}
              style={{
                height: `${height}px`,
                backgroundColor,
                left: `${node.x * 15}px`,
                bottom: `${node.y * 15}px`,
                transform: `translateZ(${height/2}px)`
              }}
              title={`(${node.x}, ${node.y}) - Curvature: ${node.curvature.toExponential(2)}`}
            />
          );
        })}
      </div>
    );
  };

  return (
    <div className="curvature-heatmap-panel">
      <div className="panel-header">
        <h2>Global Curvature Resonance</h2>
        <div className="panel-controls">
          <button 
            className={viewMode === '2D' ? 'active' : ''}
            onClick={() => setViewMode('2D')}
          >
            2D Heatmap
          </button>
          <button 
            className={viewMode === '3D' ? 'active' : ''}
            onClick={() => setViewMode('3D')}
          >
            3D Surface
          </button>
        </div>
      </div>

      {metrics && (
        <div className="metrics-display">
          <div className={`metric-item ${metrics.safe_mode ? 'critical' : 'normal'}`}>
            <span className="metric-label">Safe Mode:</span>
            <span className="metric-value">{metrics.safe_mode ? 'ACTIVE' : 'INACTIVE'}</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">RSI:</span>
            <span className="metric-value">{metrics.rsi.toFixed(3)}</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">CS:</span>
            <span className="metric-value">{metrics.cs.toFixed(3)}</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">GAS:</span>
            <span className="metric-value">{metrics.gas.toFixed(3)}</span>
          </div>
          <div className="metric-item">
            <span className="metric-label">Stability:</span>
            <span className="metric-value">{metrics.stability_state}</span>
          </div>
        </div>
      )}

      <div className="visualization-container">
        {viewMode === '2D' ? render2DHeatmap() : render3DSurface()}
      </div>

      <div className="legend">
        <div className="legend-item">
          <div className="color-box" style={{ backgroundColor: 'blue' }}></div>
          <span>Low Curvature</span>
        </div>
        <div className="legend-item">
          <div className="color-box" style={{ backgroundColor: 'red' }}></div>
          <span>High Curvature</span>
        </div>
        <div className="legend-item">
          <div className="color-box nodal-surface"></div>
          <span>Nodal Surface</span>
        </div>
      </div>
    </div>
  );
};

export default CurvatureHeatmapPanel;