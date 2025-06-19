import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import './UmapVisualization.css';
import { getUmapData } from '../../services/api';
import { formatNumber } from '../../utils/formatting';

const UmapVisualization = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedTarget, setSelectedTarget] = useState(null);
  const [colorBy, setColorBy] = useState('avg_stance');
  const [sizeBy, setSizeBy] = useState('count');
  const [filterValue, setFilterValue] = useState('');
  const [tooltip, setTooltip] = useState({
    visible: false,
    x: 0,
    y: 0,
    target: null
  });
  
  const navigate = useNavigate();
  const svgRef = useRef(null);
  const containerRef = useRef(null);
  
  // Load UMAP data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await getUmapData();
        
        if (response && response.data) {
          setData(response.data);
        } else {
          setError('No UMAP data available');
        }
      } catch (err) {
        console.error('Error fetching UMAP data:', err);
        setError('Failed to load UMAP visualization data');
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
  }, []);
  
  // Filter data based on search input
  const filteredData = useCallback(() => {
    if (!data.length) return [];
    if (!filterValue.trim()) return data;
    
    const searchTerm = filterValue.toLowerCase();
    return data.filter(item => 
      item.Target.toLowerCase().includes(searchTerm)
    );
  }, [data, filterValue]);
  
  // Calculate color scale for points
  const getColor = useCallback((item) => {
    if (!item || typeof item[colorBy] === 'undefined') return '#aaaaaa';
    
    if (colorBy === 'avg_stance') {
      // Red (negative) to blue (positive) scale for stance
      const value = item.avg_stance;
      if (value <= -0.7) return '#d32f2f';
      if (value <= -0.4) return '#f44336';
      if (value <= -0.1) return '#ffcdd2';
      if (value >= 0.7) return '#1565c0';
      if (value >= 0.4) return '#2196f3';
      if (value >= 0.1) return '#bbdefb';
      return '#e0e0e0'; // Neutral
    }
    
    if (colorBy === 'stance_abs') {
      // Gray (neutral) to purple (polarizing) scale
      const value = item.stance_abs;
      if (value >= 0.7) return '#6a1b9a';
      if (value >= 0.5) return '#9c27b0';
      if (value >= 0.3) return '#ce93d8';
      return '#e0e0e0';
    }
    
    if (colorBy === 'top_platform') {
      // Different color for each platform
      const platformColors = {
        'twitter': '#1da1f2',
        'instagram': '#c32aa3',
        'tiktok': '#000000'
      };
      return platformColors[item.top_platform] || '#aaaaaa';
    }
    
    if (colorBy === 'top_party') {
      // Different color for each party
      const partyColors = {
        'Conservative': '#0000ff',
        'Liberal': '#ff0000',
        'NDP': '#ff8c00',
        'Green': '#00ff00',
        'Bloc': '#6495ed',
        'PPC': '#800080',
        'None': '#aaaaaa'
      };
      return partyColors[item.top_party] || '#aaaaaa';
    }
    
    return '#aaaaaa';
  }, [colorBy]);
  
  // Calculate point size
  const getPointSize = useCallback((item) => {
    if (!item || typeof item[sizeBy] === 'undefined') return 5;
    
    // Base size on the selected metric
    if (sizeBy === 'count') {
      const count = item.count || 0;
      return Math.max(3, Math.min(15, 3 + Math.sqrt(count) / 10));
    }
    
    return 5; // Default size
  }, [sizeBy]);
  
  // Render the visualization
  useEffect(() => {
    if (loading || error || !data.length) return;
    
    const filtered = filteredData();
    if (!filtered.length) return;
    
    // Calculate bounds for scaling
    const padding = 50;
    const width = containerRef.current.clientWidth - padding * 2;
    const height = containerRef.current.clientHeight - padding * 2;
    
    // Find min/max values for scaling
    const xValues = filtered.map(d => d.x);
    const yValues = filtered.map(d => d.y);
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);
    
    // Scale functions
    const scaleX = (x) => (
      ((x - xMin) / (xMax - xMin)) * width + padding
    );
    
    const scaleY = (y) => (
      ((y - yMin) / (yMax - yMin)) * height + padding
    );
    
    // Clear the SVG
    const svg = svgRef.current;
    while (svg.firstChild) {
      svg.removeChild(svg.firstChild);
    }
    
    // Add points
    filtered.forEach(item => {
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      const x = scaleX(item.x);
      const y = scaleY(item.y);
      const radius = getPointSize(item);
      
      circle.setAttribute('cx', x);
      circle.setAttribute('cy', y);
      circle.setAttribute('r', radius);
      circle.setAttribute('fill', getColor(item));
      circle.setAttribute('stroke', selectedTarget === item.Target ? '#000000' : 'none');
      circle.setAttribute('stroke-width', selectedTarget === item.Target ? 2 : 0);
      circle.setAttribute('data-target', item.Target);
      
      // Interactivity
      circle.addEventListener('mouseenter', () => {
        setTooltip({
          visible: true,
          x: x + radius + 5,
          y: y,
          target: item
        });
        circle.setAttribute('stroke', '#000000');
        circle.setAttribute('stroke-width', 2);
      });
      
      circle.addEventListener('mouseleave', () => {
        setTooltip({
          ...tooltip,
          visible: false
        });
        if (selectedTarget !== item.Target) {
          circle.setAttribute('stroke', 'none');
          circle.setAttribute('stroke-width', 0);
        }
      });
      
      circle.addEventListener('click', () => {
        setSelectedTarget(item.Target);
        // Navigate to main tab with this target selected
        navigate(`/?target=${encodeURIComponent(item.Target)}`);
      });
      
      svg.appendChild(circle);
    });
    
  }, [data, loading, error, filteredData, getColor, getPointSize, selectedTarget, tooltip, navigate]);
  
  // Handle resize
  useEffect(() => {
    const handleResize = () => {
      // Trigger re-render on resize
      setData([...data]);
    };
    
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [data]);
  
  if (loading) {
    return <div className="umap-loading">Loading UMAP visualization...</div>;
  }
  
  if (error) {
    return <div className="umap-error">{error}</div>;
  }
  
  if (data.length === 0) {
    return <div className="umap-no-data">No UMAP data available</div>;
  }
  
  return (
    <div className="umap-container" ref={containerRef}>
      <div className="umap-controls">
        <div className="umap-control-group">
          <label>Color by:</label>
          <select value={colorBy} onChange={(e) => setColorBy(e.target.value)}>
            <option value="avg_stance">Average Stance</option>
            <option value="stance_abs">Polarization</option>
            <option value="top_platform">Platform</option>
            <option value="top_party">Party</option>
          </select>
        </div>
        
        <div className="umap-control-group">
          <label>Size by:</label>
          <select value={sizeBy} onChange={(e) => setSizeBy(e.target.value)}>
            <option value="count">Data point count</option>
          </select>
        </div>
        
        <div className="umap-control-group">
          <label>Filter targets:</label>
          <input 
            type="text" 
            value={filterValue} 
            onChange={(e) => setFilterValue(e.target.value)}
            placeholder="Search targets..."
          />
        </div>
      </div>
      
      <div className="umap-legend">
        {colorBy === 'avg_stance' && (
          <div className="legend-items">
            <div className="legend-item">
              <span className="color-sample" style={{ backgroundColor: '#d32f2f' }}></span>
              <span>Strongly Against</span>
            </div>
            <div className="legend-item">
              <span className="color-sample" style={{ backgroundColor: '#e0e0e0' }}></span>
              <span>Neutral</span>
            </div>
            <div className="legend-item">
              <span className="color-sample" style={{ backgroundColor: '#1565c0' }}></span>
              <span>Strongly For</span>
            </div>
          </div>
        )}
        
        {colorBy === 'stance_abs' && (
          <div className="legend-items">
            <div className="legend-item">
              <span className="color-sample" style={{ backgroundColor: '#e0e0e0' }}></span>
              <span>Neutral</span>
            </div>
            <div className="legend-item">
              <span className="color-sample" style={{ backgroundColor: '#9c27b0' }}></span>
              <span>Polarizing</span>
            </div>
          </div>
        )}
        
        {colorBy === 'top_platform' && (
          <div className="legend-items">
            <div className="legend-item">
              <span className="color-sample" style={{ backgroundColor: '#1da1f2' }}></span>
              <span>Twitter</span>
            </div>
            <div className="legend-item">
              <span className="color-sample" style={{ backgroundColor: '#c32aa3' }}></span>
              <span>Instagram</span>
            </div>
            <div className="legend-item">
              <span className="color-sample" style={{ backgroundColor: '#000000' }}></span>
              <span>TikTok</span>
            </div>
          </div>
        )}
        
        {colorBy === 'top_party' && (
          <div className="legend-items">
            <div className="legend-item">
              <span className="color-sample" style={{ backgroundColor: '#0000ff' }}></span>
              <span>Conservative</span>
            </div>
            <div className="legend-item">
              <span className="color-sample" style={{ backgroundColor: '#ff0000' }}></span>
              <span>Liberal</span>
            </div>
            <div className="legend-item">
              <span className="color-sample" style={{ backgroundColor: '#ff8c00' }}></span>
              <span>NDP</span>
            </div>
            <div className="legend-item">
              <span className="color-sample" style={{ backgroundColor: '#00ff00' }}></span>
              <span>Green</span>
            </div>
            <div className="legend-item">
              <span className="color-sample" style={{ backgroundColor: '#6495ed' }}></span>
              <span>Bloc</span>
            </div>
          </div>
        )}
      </div>
      
      <div className="umap-visualization">
        <svg ref={svgRef} width="100%" height="600"></svg>
        
        {tooltip.visible && tooltip.target && (
          <div 
            className="umap-tooltip" 
            style={{ 
              left: `${tooltip.x}px`, 
              top: `${tooltip.y}px` 
            }}
          >
            <h4>{tooltip.target.Target}</h4>
            <p>Count: {tooltip.target.count}</p>
            <p>Avg. Stance: {formatNumber(tooltip.target.avg_stance)}</p>
            <p>Polarization: {formatNumber(tooltip.target.stance_abs)}</p>
            <p>Platform: {tooltip.target.top_platform}</p>
            <p>Party: {tooltip.target.top_party}</p>
            <p className="tooltip-hint">Click to view trend</p>
          </div>
        )}
      </div>
      
      <div className="umap-description">
        <p>
          This visualization uses UMAP dimensionality reduction to show stance target relationships
          based on semantic similarity. Similar targets appear closer together. 
          Click on any point to view its trend chart.
        </p>
      </div>
    </div>
  );
};

export default UmapVisualization;