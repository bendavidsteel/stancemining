import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import DeckGL from '@deck.gl/react';
import { ScatterplotLayer } from '@deck.gl/layers';
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
  const [viewState, setViewState] = useState({
    target: [0, 0, 0],
    zoom: 0
  });
  
  const navigate = useNavigate();
  
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
  
  // Update view state when data changes
  useEffect(() => {
    if (loading || error || !data.length) return;
    
    const filtered = filteredData();
    if (!filtered.length) return;
    
    // Calculate bounds for initial view
    const xValues = filtered.map(d => d.x);
    const yValues = filtered.map(d => d.y);
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);
    
    const centerX = (xMin + xMax) / 2;
    const centerY = (yMin + yMax) / 2;
    const rangeX = xMax - xMin;
    const rangeY = yMax - yMin;
    const maxRange = Math.max(rangeX, rangeY);
    
    setViewState({
      target: [centerX, centerY, 0],
      zoom: Math.log2(600 / maxRange) - 1
    });
  }, [data, loading, error, filteredData]);
  
  // Convert color string to RGB array
  const hexToRgb = (hex) => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? [
      parseInt(result[1], 16),
      parseInt(result[2], 16),
      parseInt(result[3], 16)
    ] : [170, 170, 170];
  };
  
  // Create deck.gl layers
  const layers = [
    new ScatterplotLayer({
      id: 'umap-points',
      data: filteredData(),
      getPosition: d => [d.x, d.y],
      getRadius: d => getPointSize(d) * 50,
      getFillColor: d => {
        const color = hexToRgb(getColor(d));
        return selectedTarget === d.Target ? [...color, 255] : [...color, 200];
      },
      getLineColor: d => selectedTarget === d.Target ? [0, 0, 0, 255] : [0, 0, 0, 0],
      getLineWidth: d => selectedTarget === d.Target ? 20 : 0,
      pickable: true,
      onHover: (info) => {
        if (info.object && info.x && info.y) {
          setTooltip({
            visible: true,
            x: info.x,
            y: info.y,
            target: info.object
          });
        } else {
          setTooltip(prev => ({ ...prev, visible: false }));
        }
      },
      onClick: (info) => {
        if (info.object) {
          setSelectedTarget(info.object.Target);
          navigate(`/?target=${encodeURIComponent(info.object.Target)}`);
        }
      }
    })
  ];
  
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
    <div className="umap-container">
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
        <DeckGL
          viewState={viewState}
          onViewStateChange={({viewState}) => setViewState(viewState)}
          controller={true}
          layers={layers}
          width="100%"
          height={600}
        />
        
        {tooltip.visible && tooltip.target && (
          <div 
            className="umap-tooltip" 
            style={{ 
              left: `${tooltip.x}px`, 
              top: `${tooltip.y}px`,
              position: 'absolute',
              pointerEvents: 'none'
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