import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import Plot from 'react-plotly.js';
import './TargetChart.css';
import api, { getTargetTrendsBatch } from '../../services/api'; 

const TargetChart = ({ targetName, apiBaseUrl }) => {
  const [trendData, setTrendData] = useState([]);
  const [rawData, setRawData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filterType, setFilterType] = useState('all');
  const [filterValue, setFilterValue] = useState('all');
  const [availableFilters, setAvailableFilters] = useState({});
  const [showScatter, setShowScatter] = useState(false);
  const [availableFilterValues, setAvailableFilterValues] = useState([]);
  const [loadingRawData, setLoadingRawData] = useState(false);
  const [rawDataLoaded, setRawDataLoaded] = useState(false);
  const [multipleTimelines, setMultipleTimelines] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [plotRevision, setPlotRevision] = useState(0);
  const [chartZoomState, setChartZoomState] = useState(null);
  const [syncedXAxisRange, setSyncedXAxisRange] = useState(null);
  
  const chartContainerRef = useRef(null);

  // Generate a color for a filter value
  const getFilterColor = useCallback((filterVal, index) => {
    const colors = [
      '#8884d8', '#82ca9d', '#ffc658', '#ff8042', '#0088fe', 
      '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d'
    ];
    
    // Use index if provided, otherwise hash the string to get a consistent color
    if (index !== undefined) return colors[index % colors.length];
    
    const hashCode = filterVal.split('').reduce(
      (acc, char) => char.charCodeAt(0) + ((acc << 5) - acc), 0
    );
    return colors[Math.abs(hashCode) % colors.length];
  }, []);




  // Load available filters for this target
  useEffect(() => {
    const loadFilters = async () => {
      try {
        const response = await api.get(`/target/${targetName}/filters`);
        setAvailableFilters(response.data);
        
        // Set available filter values based on the selected filter type
        if (filterType !== 'all' && response.data[filterType]) {
          setAvailableFilterValues(response.data[filterType] || []);
        }
      } catch (err) {
        console.error('Error fetching filters:', err);
      }
    };
    
    loadFilters();
  }, [targetName, filterType]);
  
  // Load data based on current filter settings
  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        setError(null);
        // Reset raw data when filters change
        setRawDataLoaded(false);
        setShowScatter(false);
        
        // Determine if we need to load multiple timelines or a single timeline
        if (filterType !== 'all' && filterValue === 'all' && availableFilterValues.length > 0) {
          // Load all timelines in a single batch request
          const response = await getTargetTrendsBatch(targetName, filterType, availableFilterValues);
          
          // Process batch response and prepare data for chart
          const allTimelines = Object.entries(response.data).map(([value, data], index) => {
            const color = getFilterColor(value, index);
            
            const formattedData = data.map(item => {
              const timestamp = new Date(item.createtime).getTime();
              const trendMean = parseFloat(item.trend_mean) || 0;
              const trendLower = parseFloat(item.trend_lower) || 0;
              const trendUpper = parseFloat(item.trend_upper) || 0;
              return {
                x: timestamp,
                [`trend_mean_${value}`]: trendMean,
                [`trend_lower_${value}`]: trendLower,
                [`trend_upper_${value}`]: trendUpper,
                [`ci_base_${value}`]: trendLower,
                [`ci_fill_${value}`]: trendUpper - trendLower,
                [`volume_${value}`]: parseInt(item.volume) || 0
              };
            });
            
            return { filterValue: value, data: formattedData, color };
          });
          
          // Combine data for the chart
          const combinedDataMap = new Map();
          
          allTimelines.forEach(timeline => {
            timeline.data.forEach(point => {
              if (!combinedDataMap.has(point.x)) {
                combinedDataMap.set(point.x, { x: point.x });
              }
              
              const existingPoint = combinedDataMap.get(point.x);
              existingPoint[`trend_mean_${timeline.filterValue}`] = point[`trend_mean_${timeline.filterValue}`];
              existingPoint[`trend_lower_${timeline.filterValue}`] = point[`trend_lower_${timeline.filterValue}`];
              existingPoint[`trend_upper_${timeline.filterValue}`] = point[`trend_upper_${timeline.filterValue}`];
              existingPoint[`ci_base_${timeline.filterValue}`] = point[`ci_base_${timeline.filterValue}`];
              existingPoint[`ci_fill_${timeline.filterValue}`] = point[`ci_fill_${timeline.filterValue}`];
              existingPoint[`volume_${timeline.filterValue}`] = point[`volume_${timeline.filterValue}`];
            });
          });
          
          const combinedData = Array.from(combinedDataMap.values())
            .sort((a, b) => a.x - b.x);
          
          setMultipleTimelines(allTimelines);
          setTrendData(combinedData);
          setPlotRevision(prev => prev + 1);
        } else {
          // Load a single timeline
          const response = await api.get(`/target/${targetName}/trends`, {
            params: { filter_type: filterType, filter_value: filterValue }
          });
          
          if (!response.data.data || response.data.data.length === 0) {
            setTrendData([]);
            return;
          }
          
          const formattedData = response.data.data.map(item => {
            const timestamp = new Date(item.createtime).getTime();
            const trendMean = parseFloat(item.trend_mean) || 0;
            const trendLower = parseFloat(item.trend_lower) || 0;
            const trendUpper = parseFloat(item.trend_upper) || 0;
            return {
              x: timestamp,
              trend_mean: trendMean,
              trend_lower: trendLower,
              trend_upper: trendUpper,
              ci_base: trendLower,
              ci_fill: trendUpper - trendLower,
              volume: parseInt(item.volume) || 0
            };
          });
          
          setTrendData(formattedData);
          setPlotRevision(prev => prev + 1);
        }
      } catch (err) {
        console.error('Error loading chart data:', err);
        setError('Failed to load chart data');
      } finally {
        setLoading(false);
      }
    };
    
    loadData();
  }, [targetName, filterType, filterValue, availableFilterValues, getFilterColor]);
  
  // Capture current zoom and pan state from chart
  const captureZoomState = () => {
    const chartElement = chartContainerRef.current?.querySelector('.js-plotly-plot');
    if (chartElement && chartElement.layout) {
      const layout = chartElement.layout;
      return {
        xaxis: layout.xaxis ? { 
          range: layout.xaxis.range,
          autorange: layout.xaxis.autorange
        } : null,
        yaxis: layout.yaxis ? { 
          range: layout.yaxis.range,
          autorange: layout.yaxis.autorange
        } : null
      };
    }
    return null;
  };

  // Toggle scatter plot (raw data points)
  const toggleScatter = async () => {
    if (!showScatter) {
      // Capture current zoom state before loading data
      const currentZoomState = captureZoomState();
      setChartZoomState(currentZoomState);
      
      // Always fetch fresh data when showing scatter points
      try {
        setLoadingRawData(true);
        
        const response = await api.get(`/target/${targetName}/raw`, {
          params: { filter_type: filterType, filter_value: filterValue }
        });
        
        const formattedData = response.data.data.map(item => ({
          ...item,
          x: new Date(item.createtime).getTime(),
          Stance: parseFloat(item.Stance) || 0,
          hoverText: Object.entries(item)
            .filter(([key]) => key !== 'createtime') // Exclude createtime as we show formatted date
            .map(([key, value]) => `${key}: ${value}`)
            .join('<br>')
        }));
        
        setRawData(formattedData);
        setRawDataLoaded(true);
        setShowScatter(true);
      } catch (err) {
        console.error('Error loading raw data:', err);
      } finally {
        setLoadingRawData(false);
      }
    } else {
      setShowScatter(false);
    }
  };
  
  // Handle filter type change
  const handleFilterTypeChange = (e) => {
    const newType = e.target.value;
    // Reset filter value and search term when changing type
    setFilterValue('all');
    setSearchTerm('');
    setShowSuggestions(false);
    setFilterType(newType);
  };
  
  
  // Handle search input change
  const handleSearchChange = (e) => {
    const value = e.target.value;
    setSearchTerm(value);
    setShowSuggestions(value.length > 0);
  };
  
  // Handle suggestion selection
  const handleSuggestionSelect = (value) => {
    setFilterValue(value);
    setSearchTerm(value);
    setShowSuggestions(false);
  };
  
  // Filter suggestions based on search term
  const filteredSuggestions = availableFilterValues.filter(value => 
    value.toLowerCase().includes(searchTerm.toLowerCase())
  ).slice(0, 10); // Limit to 10 suggestions
  

  // Handle relayout events to sync zoom/pan between charts
  const handleRelayout = (eventData) => {
    if (eventData['xaxis.range[0]'] && eventData['xaxis.range[1]']) {
      setSyncedXAxisRange([eventData['xaxis.range[0]'], eventData['xaxis.range[1]']]);
    } else if (eventData['xaxis.autorange']) {
      setSyncedXAxisRange(null);
    }
  };

  // Render multiple timelines chart with confidence intervals
  const renderMultipleTimelinesChart = () => {
    const traces = [];
    const usesSingleColor = availableFilterValues.length > 5;
    const limitedFilterValues = usesSingleColor ? availableFilterValues : availableFilterValues.slice(0, 5);
    
    // Calculate alpha based on number of lines (more lines = lower alpha)
    const baseAlpha = usesSingleColor ? Math.max(0.3, 1 - (availableFilterValues.length * 0.05)) : 1;
    const fillAlpha = usesSingleColor ? Math.max(0.01, 1 - (availableFilterValues.length * 0.05)) : 0.3;
    
    // Create traces for each filter value
    limitedFilterValues.forEach((filterVal, index) => {
      const color = usesSingleColor ? '#8884d8' : getFilterColor(filterVal, index);
      const xValues = trendData.map(d => new Date(d.x));
      
      // Confidence interval fill
      const upperValues = trendData.map(d => d[`trend_upper_${filterVal}`]);
      const lowerValues = trendData.map(d => d[`trend_lower_${filterVal}`]);
      
      // Add upper bound trace (invisible)
      traces.push({
        x: xValues,
        y: upperValues,
        type: 'scatter',
        mode: 'lines',
        line: { color: 'transparent' },
        showlegend: false,
        hoverinfo: 'skip',
        name: `upper_${filterVal}`
      });
      
      // Add confidence interval fill
      const rgbColor = usesSingleColor 
        ? `136, 132, 216` // RGB for #8884d8
        : `${parseInt(color.slice(1, 3), 16)}, ${parseInt(color.slice(3, 5), 16)}, ${parseInt(color.slice(5, 7), 16)}`;
      
      traces.push({
        x: xValues,
        y: lowerValues,
        type: 'scatter',
        mode: 'lines',
        line: { color: 'transparent' },
        fill: 'tonexty',
        fillcolor: `rgba(${rgbColor}, ${fillAlpha})`,
        showlegend: false,
        hoverinfo: 'skip',
        name: `fill_${filterVal}`
      });
      
      // Add main trend line
      const lineColor = usesSingleColor 
        ? `rgba(136, 132, 216, ${baseAlpha})`
        : color;
      
      traces.push({
        x: xValues,
        y: trendData.map(d => d[`trend_mean_${filterVal}`]),
        type: 'scatter',
        mode: 'lines',
        line: { color: lineColor, width: 2 },
        name: filterVal,
        showlegend: !usesSingleColor,
        hovertemplate: `<b>${filterVal}</b><br>Date: %{x}<br>Stance: %{y:.2f}<extra></extra>`
      });
    });
    
    // Volume chart traces
    const volumeTraces = limitedFilterValues.map((filterVal, index) => {
      const volumeColor = usesSingleColor 
        ? `rgba(136, 132, 216, ${baseAlpha})`
        : getFilterColor(filterVal, index);
      
      return {
        x: trendData.map(d => new Date(d.x)),
        y: trendData.map(d => d[`volume_${filterVal}`]),
        type: 'scatter',
        mode: 'lines',
        line: { color: volumeColor },
        name: filterVal,
        showlegend: !usesSingleColor,
        hovertemplate: `<b>${filterVal}</b><br>Date: %{x}<br>Volume: %{y}<extra></extra>`
      };
    });
    
    // Create shared x-axis layout
    const sharedXAxisLayout = {
      title: 'Date',
      type: 'date',
      ...(syncedXAxisRange && { range: syncedXAxisRange })
    };
    
    return (
      <>
        <div className="stance-chart">
          <Plot
            data={traces}
            layout={{
              title: 'Stance Over Time',
              xaxis: sharedXAxisLayout,
              yaxis: { 
                title: 'Stance',
                range: [-1, 1],
                tickvals: [-1, -0.5, 0, 0.5, 1],
                ticktext: ['Against', '', 'Neutral', '', 'For'],
                fixedrange: true
              },
              height: 300,
              margin: { l: 60, r: 40, t: 40, b: 40 },
              showlegend: !usesSingleColor,
              legend: usesSingleColor ? {} : { orientation: 'h', y: -0.2 },
            }}
            config={{ responsive: true, displayModeBar: true }}
            style={{ width: '100%', height: '300px' }}
            revision={plotRevision}
            onRelayout={handleRelayout}
          />
        </div>
        
        <div className="volume-chart">
          <Plot
            data={volumeTraces}
            layout={{
              title: 'Volume Over Time',
              xaxis: sharedXAxisLayout,
              yaxis: { title: 'Volume', fixedrange: true },
              height: 150,
              margin: { l: 60, r: 40, t: 40, b: 40 },
              showlegend: !usesSingleColor,
              legend: usesSingleColor ? {} : { orientation: 'h', y: -0.4 }
            }}
            config={{ responsive: true, displayModeBar: true }}
            style={{ width: '100%', height: '150px' }}
            revision={plotRevision}
            onRelayout={handleRelayout}
          />
        </div>
      </>
    );
  };

  // Render single timeline chart
  const renderSingleTimelineChart = () => {
    const xValues = trendData.map(d => new Date(d.x));
    const upperValues = trendData.map(d => d.trend_upper);
    const lowerValues = trendData.map(d => d.trend_lower);
    const meanValues = trendData.map(d => d.trend_mean);
    const volumeValues = trendData.map(d => d.volume);
    
    const traces = [
      // Upper bound (invisible)
      {
        x: xValues,
        y: upperValues,
        type: 'scatter',
        mode: 'lines',
        line: { color: 'transparent' },
        showlegend: false,
        hoverinfo: 'skip',
        name: 'upper'
      },
      // Confidence interval fill
      {
        x: xValues,
        y: lowerValues,
        type: 'scatter',
        mode: 'lines',
        line: { color: 'transparent' },
        fill: 'tonexty',
        fillcolor: 'rgba(136, 132, 216, 0.3)',
        showlegend: false,
        hoverinfo: 'skip',
        name: 'confidence'
      },
      // Main trend line
      {
        x: xValues,
        y: meanValues,
        type: 'scatter',
        mode: 'lines',
        line: { color: '#8884d8', width: 2 },
        name: 'Average stance',
        hovertemplate: '<b>Average stance</b><br>Date: %{x}<br>Stance: %{y:.2f}<extra></extra>'
      }
    ];
    
    // Add scatter points if enabled
    if (showScatter && rawDataLoaded) {
      traces.push({
        x: rawData.map(d => new Date(d.x)),
        y: rawData.map(d => d.Stance),
        type: 'scatter',
        mode: 'markers',
        marker: { color: '#1E90FF', opacity: 0.5, size: 4 },
        name: 'Data points',
        text: rawData.map(d => d.hoverText),
        hovertemplate: '<b>Data point</b><br>Date: %{x}<br>%{text}<extra></extra>'
      });
    }
    
    // Create shared x-axis layout
    const sharedXAxisLayout = {
      title: 'Date',
      type: 'date',
      ...(syncedXAxisRange && { range: syncedXAxisRange })
    };
    
    return (
      <>
        <div className="stance-chart">
          <Plot
            data={traces}
            layout={{
              title: 'Stance Over Time',
              xaxis: sharedXAxisLayout,
              yaxis: { 
                title: 'Stance',
                range: [-1, 1],
                tickvals: [-1, -0.5, 0, 0.5, 1],
                ticktext: ['Against', '', 'Neutral', '', 'For'],
                fixedrange: true
              },
              height: 200,
              margin: { l: 60, r: 40, t: 40, b: 40 },
              showlegend: true,
              legend: { orientation: 'h', y: -0.3 },
            }}
            config={{ responsive: true, displayModeBar: true }}
            style={{ width: '100%', height: '200px' }}
            revision={plotRevision}
            onRelayout={handleRelayout}
            onUpdate={(figure) => {
              if (chartZoomState && figure.layout) {
                if (chartZoomState.xaxis?.range) {
                  figure.layout.xaxis.range = chartZoomState.xaxis.range;
                  figure.layout.xaxis.autorange = chartZoomState.xaxis.autorange;
                }
                if (chartZoomState.yaxis?.range) {
                  figure.layout.yaxis.range = chartZoomState.yaxis.range;
                  figure.layout.yaxis.autorange = chartZoomState.yaxis.autorange;
                }
                setChartZoomState(null); // Clear after applying
              }
            }}
          />
        </div>
        
        <div className="volume-chart">
          <Plot
            data={[{
              x: xValues,
              y: volumeValues,
              type: 'scatter',
              mode: 'lines',
              fill: 'tozeroy',
              fillcolor: 'rgba(130, 202, 157, 0.3)',
              line: { color: '#82ca9d' },
              name: 'Volume',
              hovertemplate: '<b>Volume</b><br>Date: %{x}<br>Volume: %{y}<extra></extra>'
            }]}
            layout={{
              title: 'Volume Over Time',
              xaxis: sharedXAxisLayout,
              yaxis: { title: 'Volume', fixedrange: true },
              height: 100,
              margin: { l: 60, r: 40, t: 40, b: 40 },
              showlegend: false
            }}
            config={{ responsive: true, displayModeBar: true }}
            style={{ width: '100%', height: '100px' }}
            revision={plotRevision}
            onRelayout={handleRelayout}
          />
        </div>
      </>
    );
  };

  if (loading) {
    return <div className="target-chart loading">Loading chart data...</div>;
  }

  if (error) {
    return <div className="target-chart error">{error}</div>;
  }

  if (trendData.length === 0) {
    // Show loading state if we're currently loading data
    if (loading) {
      return (
        <div className="target-chart loading">
          <h2>{targetName}</h2>
          <div className="loading-bar">
            <div className="loading-bar-fill"></div>
          </div>
          <p>Loading chart data...</p>
        </div>
      );
    }
    
    // Show no data message only when not loading
    return (
      <div className="target-chart no-data">
        <h2>{targetName}</h2>
        <p>No data available for this target</p>
      </div>
    );
  }

  const showingMultipleTimelines = filterType !== 'all' && filterValue === 'all';

  return (
    <div className="target-chart">
      <h2>{targetName}</h2>
      
      <div className="chart-controls">
        <div className="filter-controls">
          <div className="filter-group">
            <label>Filter by:</label>
            <select value={filterType} onChange={handleFilterTypeChange}>
              {Object.keys(availableFilters).sort().map(filterKey => (
                <option key={filterKey} value={filterKey}>
                  {filterKey}
                </option>
              ))}
            </select>
          </div>
          
          {filterType !== 'all' && (
            <div className="filter-group searchable-filter">
              <label>Select value:</label>
              <div className="search-container">
                <input
                  type="text"
                  value={searchTerm}
                  onChange={handleSearchChange}
                  onFocus={() => setShowSuggestions(searchTerm.length > 0)}
                  onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
                  placeholder={filterValue === 'all' ? 'Search for a value...' : `Current: ${filterValue}`}
                  disabled={filterType === 'all'}
                />
                {showSuggestions && filteredSuggestions.length > 0 && (
                  <div className="suggestions-dropdown">
                    {filteredSuggestions.map(value => (
                      <div 
                        key={value} 
                        className="suggestion-item"
                        onClick={() => handleSuggestionSelect(value)}
                      >
                        {value}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
        
        <div className="chart-actions">
          {!showingMultipleTimelines && (
            <div className="scatter-toggle">
              <button
                type="button"
                className={`chart-button ${showScatter ? 'active' : ''}`}
                onClick={toggleScatter}
                disabled={loadingRawData}
              >
                {loadingRawData ? 'Loading data points...' : (showScatter ? 'Hide data points' : 'Show data points')}
              </button>
            </div>
          )}
          
        </div>
      </div>
      
      <div 
        ref={chartContainerRef} 
        className="charts-container"
        style={{ position: 'relative' }}
      >
        {loadingRawData && (
          <div className="chart-loading-overlay">
            <div className="loading-spinner"></div>
            <span>Loading data points...</span>
          </div>
        )}
        {showingMultipleTimelines ? 
          renderMultipleTimelinesChart() : 
          renderSingleTimelineChart()
        }
      </div>
      
      <div className="chart-instructions">
        <p>
          <strong>Zoom/Pan:</strong> Use Plotly controls for zoom and pan.
          {showingMultipleTimelines && <span> <strong>Note:</strong> Select 'All {filterType}s' to compare timelines together.</span>}
        </p>
      </div>
    </div>
  );
};

export default TargetChart;