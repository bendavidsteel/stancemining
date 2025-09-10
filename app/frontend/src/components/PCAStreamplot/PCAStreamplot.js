import React, { useState, useEffect, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { getPcaStreamplotData, handleApiError } from '../../services/api';
import './PCAStreamplot.css';

const PCAStreamplot = () => {
  const [data, setData] = useState([]);
  const [pcaInfo, setPcaInfo] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [viewMode, setViewMode] = useState('all'); // 'all', 'by-year', 'by-platform'
  const [selectedYear, setSelectedYear] = useState('all');
  const [selectedPlatform, setSelectedPlatform] = useState('all');
  const [showFlowField, setShowFlowField] = useState(true);
  const [showDensity, setShowDensity] = useState(true);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      const result = await getPcaStreamplotData();
      setData(result.data || []);
      setPcaInfo(result.pca_info || {});
      setLoading(false);
    } catch (err) {
      console.error('Error fetching PCA data:', err);
      setError(handleApiError(err));
      setLoading(false);
    }
  };

  // Get unique years and platforms for filtering
  const { uniqueYears, uniquePlatforms } = useMemo(() => {
    const years = [...new Set(data.map(d => d.year))].sort();
    const platforms = [...new Set(data.map(d => d.platform))].sort();
    return { uniqueYears: years, uniquePlatforms: platforms };
  }, [data]);

  // Filter data based on current selection
  const filteredData = useMemo(() => {
    let filtered = data;

    if (selectedYear !== 'all') {
      filtered = filtered.filter(d => d.year === parseInt(selectedYear));
    }

    if (selectedPlatform !== 'all') {
      filtered = filtered.filter(d => d.platform === selectedPlatform);
    }

    return filtered;
  }, [data, selectedYear, selectedPlatform]);

  // Create Plotly traces
  const plotData = useMemo(() => {
    if (filteredData.length === 0) return [];

    const traces = [];

    // Create density trace using histogram2d
    if (showDensity && filteredData.length > 0) {
      const densityTrace = {
        x: filteredData.map(d => d.x),
        y: filteredData.map(d => d.y),
        type: 'histogram2d',
        colorscale: 'Viridis',
        opacity: 0.7,
        showscale: false,
        name: 'Stance Density',
        hovertemplate: 'Density: %{z}<extra></extra>',
      };
      traces.push(densityTrace);
    }

    // Create scatter trace for data points
    const scatterTrace = {
      x: filteredData.map(d => d.x),
      y: filteredData.map(d => d.y),
      type: 'scatter',
      mode: 'markers',
      marker: {
        size: 4,
        color: filteredData.map(d => d.year),
        colorscale: 'Plasma',
        opacity: 0.6,
        colorbar: {
          title: 'Year',
        },
      },
      text: filteredData.map(d => 
        `Platform: ${d.platform}<br>Year: ${d.year}<br>Date: ${d.createtime}<br>User: ${d.filter_value}`
      ),
      hovertemplate: '%{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>',
      name: 'Data Points',
    };
    traces.push(scatterTrace);

    // Add flow field arrows if available
    if (showFlowField && pcaInfo.flow_data && pcaInfo.flow_data.length > 0) {
      const flowData = pcaInfo.flow_data;
      
      // Create arrow traces
      const arrowTrace = {
        x: [],
        y: [],
        type: 'scatter',
        mode: 'lines',
        line: {
          color: 'red',
          width: 1.5,
        },
        opacity: 0.7,
        showlegend: false,
        hoverinfo: 'skip',
      };

      // Create arrows by drawing lines from start to end points
      flowData.slice(0, 200).forEach(flow => { // Limit arrows for performance
        if (Math.sqrt(flow.u * flow.u + flow.v * flow.v) > 0.05) { // Only show significant flows
          arrowTrace.x.push(flow.x, flow.x + flow.u, null);
          arrowTrace.y.push(flow.y, flow.y + flow.v, null);
        }
      });

      if (arrowTrace.x.length > 0) {
        traces.push(arrowTrace);
      }
    }

    return traces;
  }, [filteredData, pcaInfo, showDensity, showFlowField]);

  // Format axis labels with PCA component info
  const getAxisLabel = (componentIndex) => {
    if (!pcaInfo.explained_variance_ratio || !pcaInfo.components || !pcaInfo.feature_names) {
      return `PC${componentIndex + 1}`;
    }

    const variance = (pcaInfo.explained_variance_ratio[componentIndex] * 100).toFixed(1);
    const component = pcaInfo.components[componentIndex];
    
    // Get top contributing features
    const featureContributions = component
      .map((loading, idx) => ({ feature: pcaInfo.feature_names[idx], loading: Math.abs(loading) }))
      .sort((a, b) => b.loading - a.loading)
      .slice(0, 3);

    const topFeatures = featureContributions
      .map(f => f.feature.replace('trend_mean_', '').replace('volume_', ''))
      .join(', ');

    return `PC${componentIndex + 1} (${variance}%)<br>Top features: ${topFeatures}`;
  };

  const plotLayout = {
    title: {
      text: 'Interactive PCA Stance Landscape with Flow Patterns',
      font: { size: 18 },
    },
    xaxis: {
      title: getAxisLabel(0),
      titlefont: { size: 12 },
    },
    yaxis: {
      title: getAxisLabel(1),
      titlefont: { size: 12 },
    },
    hovermode: 'closest',
    showlegend: true,
    legend: {
      orientation: 'h',
      y: -0.2,
    },
    margin: { t: 80, b: 100, l: 80, r: 40 },
  };

  const config = {
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
    toImageButtonOptions: {
      format: 'png',
      filename: 'pca_stance_landscape',
      height: 800,
      width: 1200,
      scale: 1
    }
  };

  if (loading) {
    return (
      <div className="pca-streamplot-container">
        <div className="loading">Loading PCA analysis...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="pca-streamplot-container">
        <div className="error">Error: {error}</div>
      </div>
    );
  }

  return (
    <div className="pca-streamplot-container">
      <div className="controls">
        <div className="control-group">
          <label>View Mode:</label>
          <select value={viewMode} onChange={(e) => setViewMode(e.target.value)}>
            <option value="all">All Data</option>
            <option value="by-year">By Year</option>
            <option value="by-platform">By Platform</option>
          </select>
        </div>

        {(viewMode === 'all' || viewMode === 'by-year') && (
          <div className="control-group">
            <label>Year:</label>
            <select value={selectedYear} onChange={(e) => setSelectedYear(e.target.value)}>
              <option value="all">All Years</option>
              {uniqueYears.map(year => (
                <option key={year} value={year}>{year}</option>
              ))}
            </select>
          </div>
        )}

        {(viewMode === 'all' || viewMode === 'by-platform') && (
          <div className="control-group">
            <label>Platform:</label>
            <select value={selectedPlatform} onChange={(e) => setSelectedPlatform(e.target.value)}>
              <option value="all">All Platforms</option>
              {uniquePlatforms.map(platform => (
                <option key={platform} value={platform}>{platform}</option>
              ))}
            </select>
          </div>
        )}

        <div className="control-group">
          <label>
            <input
              type="checkbox"
              checked={showDensity}
              onChange={(e) => setShowDensity(e.target.checked)}
            />
            Show Density
          </label>
        </div>

        <div className="control-group">
          <label>
            <input
              type="checkbox"
              checked={showFlowField}
              onChange={(e) => setShowFlowField(e.target.checked)}
            />
            Show Flow Field
          </label>
        </div>

        <div className="data-info">
          Showing {filteredData.length} data points
          {pcaInfo.explained_variance_ratio && (
            <div className="pca-info">
              Total variance explained: {(pcaInfo.explained_variance_ratio.reduce((a, b) => a + b, 0) * 100).toFixed(1)}%
            </div>
          )}
        </div>
      </div>

      <div className="plot-container">
        <Plot
          data={plotData}
          layout={plotLayout}
          config={config}
          style={{ width: '100%', height: '700px' }}
          useResizeHandler
        />
      </div>

      <div className="explanation">
        <h3>About this Visualization</h3>
        <p>
          This interactive plot shows a 2D projection of high-dimensional stance data using Principal Component Analysis (PCA).
          Each point represents a stance measurement at a specific time and platform. The density heatmap shows where
          stance states cluster together, while flow arrows indicate how user stances evolve over time.
        </p>
        <ul>
          <li><strong>Density:</strong> Darker regions indicate more common stance positions</li>
          <li><strong>Flow Arrows:</strong> Show the direction of stance movement over time</li>
          <li><strong>Colors:</strong> Points are colored by year to show temporal patterns</li>
          <li><strong>Hover:</strong> Mouse over points to see detailed information</li>
        </ul>
      </div>
    </div>
  );
};

export default PCAStreamplot;