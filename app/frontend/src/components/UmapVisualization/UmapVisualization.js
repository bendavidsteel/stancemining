import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import Plot from 'react-plotly.js';
import './UmapVisualization.css';
import { getUmapData } from '../../services/api';
import { formatNumber } from '../../utils/formatting';

const UmapVisualization = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedTarget, setSelectedTarget] = useState(null);
  const [colorBy, setColorBy] = useState('mean_stance');
  const [sizeBy, setSizeBy] = useState('count');
  const [filterValue, setFilterValue] = useState('');
  
  
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
  const filteredData = useMemo(() => {
    if (!data.length) return [];
    if (!filterValue.trim()) return data;
    
    const searchTerm = filterValue.toLowerCase();
    return data.filter(item => 
      item && item.Target && item.Target.toLowerCase().includes(searchTerm)
    );
  }, [data, filterValue]);
  
  
  // Calculate color scale for points
  const getColor = useCallback((item) => {
    if (!item || typeof item[colorBy] === 'undefined') return '#aaaaaa';
    
    if (colorBy === 'mean_stance') {
      // Red (negative) to blue (positive) scale for stance
      const value = item.mean_stance;
      if (value <= -0.7) return '#d32f2f';
      if (value <= -0.4) return '#f44336';
      if (value <= -0.1) return '#ffcdd2';
      if (value >= 0.7) return '#15c065';
      if (value >= 0.4) return '#21f396';
      if (value >= 0.1) return '#bbfbde';
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
    
    return '#aaaaaa';
  }, [colorBy]);
  
  // Calculate point size for Plotly
  const getPointSize = (item) => {
    if (!item || typeof item[sizeBy] === 'undefined') return 8;
    
    if (sizeBy === 'count') {
      const count = item.count || 0;
      return Math.max(4, Math.min(20, 4 + Math.sqrt(count) / 5));
    }
    
    return 8;
  };
  
  
  
  // Prepare data for Plotly
  const plotlyData = useMemo(() => {
    const x = [];
    const y = [];
    const colors = [];
    const sizes = [];
    const text = [];
    const customdata = [];
    
    filteredData.forEach(item => {
      x.push(parseFloat(item.x) || 0);
      y.push(parseFloat(item.y) || 0);
      colors.push(getColor(item));
      sizes.push(getPointSize(item));
      text.push(
        `<b>${item.Target}</b><br>` +
        `Count: ${item.count}<br>` +
        `Avg. Stance: ${formatNumber(item.avg_stance)}<br>` +
        `Polarization: ${formatNumber(item.stance_abs)}<br>` +
        `<i>Click to view trend</i>`
      );
      customdata.push(item);
    });
    
    return [{
      x,
      y,
      mode: 'markers',
      type: 'scatter',
      marker: {
        color: colors,
        size: sizes,
        line: {
          color: filteredData.map(item => 
            selectedTarget === item.Target ? '#000000' : 'rgba(0,0,0,0)'
          ),
          width: 2
        },
        opacity: 0.8
      },
      text,
      customdata,
      hovertemplate: '%{text}<extra></extra>',
      hoverlabel: {
        bgcolor: 'rgba(255,255,255,0.95)',
        bordercolor: '#ddd',
        font: { size: 12 }
      }
    }];
  }, [filteredData, selectedTarget, colorBy, sizeBy, getColor, getPointSize, formatNumber]);
  
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
            <option value="mean_stance">Mean Stance</option>
            <option value="stance_abs">Polarization</option>
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
        {colorBy === 'mean_stance' && (
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
              <span className="color-sample" style={{ backgroundColor: '#15c065' }}></span>
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
      </div>
      
      <div className="umap-visualization">
        <Plot
          data={plotlyData}
          layout={{
            showlegend: false,
            hovermode: 'closest',
            xaxis: {
              title: 'UMAP Dimension 1',
              showgrid: true,
              zeroline: false
            },
            yaxis: {
              title: 'UMAP Dimension 2',
              showgrid: true,
              zeroline: false
            },
            plot_bgcolor: '#f9f9f9',
            paper_bgcolor: 'white',
            margin: { l: 50, r: 20, t: 20, b: 50 },
            autosize: true
          }}
          style={{
            width: '100%',
            height: '600px'
          }}
          config={{
            displayModeBar: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
            displaylogo: false,
            toImageButtonOptions: {
              format: 'png',
              filename: 'umap_visualization',
              height: 600,
              width: 1000,
              scale: 1
            }
          }}
          onClick={(event) => {
            if (event.points && event.points.length > 0) {
              const point = event.points[0];
              const item = point.customdata;
              if (item) {
                setSelectedTarget(item.Target);
                navigate(`/?target=${encodeURIComponent(item.Target)}`);
              }
            }
          }}
        />
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