import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import {
  ResponsiveContainer,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ComposedChart,
  Line,
  Scatter,
  AreaChart,
  Legend
} from 'recharts';
import './TargetChart.css';
import api from '../../services/api'; 

const TargetChart = ({ targetName, apiBaseUrl }) => {
  const [trendData, setTrendData] = useState([]);
  const [rawData, setRawData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filterType, setFilterType] = useState('all');
  const [filterValue, setFilterValue] = useState('all');
  const [availableFilters, setAvailableFilters] = useState({
    platforms: [],
    parties: []
  });
  const [showScatter, setShowScatter] = useState(false);
  const [availableFilterValues, setAvailableFilterValues] = useState([]);
  const [loadingRawData, setLoadingRawData] = useState(false);
  const [rawDataLoaded, setRawDataLoaded] = useState(false);
  const [multipleTimelines, setMultipleTimelines] = useState([]);
  
  const [zoomLevel, setZoomLevel] = useState(1);
  const [panOffset, setPanOffset] = useState(0);
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState(null);
  const [panStartOffset, setPanStartOffset] = useState(0);
  const [fullDomain, setFullDomain] = useState(null);
  
  // Refs for charts to sync zoom
  const stanceChartRef = useRef(null);
  const volumeChartRef = useRef(null);
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

  // Format date for display - now with day number when zoomed
  const formatAxisDate = (date) => {
    if (!date) return '';
    try {
      const dateObj = typeof date === 'number' ? new Date(date) : new Date(date);
      // Add day numbers when zoomed in
      if (zoomLevel > 2) {
        return dateObj.toLocaleDateString(undefined, {
          month: 'short',
          day: 'numeric',
          year: 'numeric'
        });
      } else {
        return dateObj.toLocaleDateString(undefined, {
          month: 'short',
          year: 'numeric'
        });
      }
    } catch (error) {
      return '';
    }
  };
  
  // Full date format for tooltips
  const formatTooltipDate = (date) => {
    if (!date) return '';
    try {
      const dateObj = typeof date === 'number' ? new Date(date) : new Date(date);
      return dateObj.toLocaleDateString(undefined, {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
      });
    } catch (error) {
      return String(date);
    }
  };

  // Calculate visible domain for chart (for zooming/panning)
  const visibleDomain = useMemo(() => {
    if (!fullDomain || zoomLevel === 1) {
      return fullDomain;
    }
    
    const domainRange = fullDomain[1] - fullDomain[0];
    const visibleRange = domainRange / zoomLevel;
    const midPoint = (fullDomain[0] + fullDomain[1]) / 2;
    const pannedMidPoint = midPoint + panOffset;
    
    return [
      pannedMidPoint - visibleRange / 2,
      pannedMidPoint + visibleRange / 2
    ];
  }, [fullDomain, zoomLevel, panOffset]);

  // Handle mouse wheel for zooming
  const handleWheel = useCallback((event) => {
    if (!fullDomain) return;
    
    event.preventDefault();
    const zoomFactor = event.deltaY < 0 ? 1.2 : 0.8;
    const newZoomLevel = Math.max(1, Math.min(10, zoomLevel * zoomFactor));
    
    if (newZoomLevel !== zoomLevel) {
      setZoomLevel(newZoomLevel);
    }
  }, [fullDomain, zoomLevel]);

  // Handle mouse down for panning
  const handleMouseDown = useCallback((e) => {
    if (zoomLevel > 1) {
      setIsPanning(true);
      setPanStart(e.clientX);
      setPanStartOffset(panOffset);
    }
  }, [zoomLevel, panOffset]);

  // Handle mouse move for panning
  const handleMouseMove = useCallback((e) => {
    if (isPanning && panStart !== null && fullDomain) {
      const deltaX = e.clientX - panStart;
      
      // Calculate pan amount based on domain range and chart width
      const domainRange = fullDomain[1] - fullDomain[0];
      const panRatio = deltaX / 500; // Approximate chart width factor
      const totalPanAmount = -panRatio * (domainRange / zoomLevel) * 0.5;
      
      // Calculate new offset from the original starting position
      const newOffset = panStartOffset + totalPanAmount;
      
      // Calculate boundaries to prevent panning too far
      const maxPanRange = (domainRange * (1 - 1/zoomLevel)) / 2;
      
      // Clamp pan offset within bounds and set directly
      const clampedOffset = Math.max(-maxPanRange, Math.min(maxPanRange, newOffset));
      setPanOffset(clampedOffset);
    }
  }, [isPanning, panStart, panStartOffset, fullDomain, zoomLevel]);

  // Handle mouse up to end panning
  const handleMouseUp = useCallback(() => {
    setIsPanning(false);
    setPanStart(null);
    setPanStartOffset(0);
  }, []);
  
  // Reset zoom
  const handleResetZoom = () => {
    setZoomLevel(1);
    setPanOffset(0);
  };
  
  // Attach wheel event listener
  useEffect(() => {
    const container = chartContainerRef.current;
    if (container) {
      container.addEventListener('wheel', handleWheel, { passive: false });
      return () => {
        container.removeEventListener('wheel', handleWheel);
      };
    }
  }, [handleWheel]);
  
  // Add global mouse move and up listeners when panning
  useEffect(() => {
    if (isPanning) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      
      return () => {
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isPanning, handleMouseMove, handleMouseUp]);

  // Load available filters for this target
  useEffect(() => {
    const loadFilters = async () => {
      try {
        const response = await api.get(`/target/${targetName}/filters`);
        setAvailableFilters(response.data);
        
        // Set available filter values based on the selected filter type
        if (filterType === 'platform') {
          setAvailableFilterValues(response.data.platforms || []);
        } else if (filterType === 'party') {
          setAvailableFilterValues(response.data.parties || []);
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
        
        // Determine if we need to load multiple timelines or a single timeline
        if (filterType !== 'all' && filterValue === 'all' && availableFilterValues.length > 0) {
          // Load multiple timelines for comparison
          const promises = availableFilterValues.map(value => 
            api.get(`/target/${targetName}/trends`, {
              params: { filter_type: filterType, filter_value: value }
            })
          );
          
          const responses = await Promise.all(promises);
          
          // Process responses and prepare data for chart
          const allTimelines = responses.map((response, index) => {
            const value = availableFilterValues[index];
            const color = getFilterColor(value, index);
            
            const formattedData = response.data.data.map(item => {
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
          
          if (combinedData.length > 0) {
            const timestamps = combinedData.map(d => d.x);
            setFullDomain([Math.min(...timestamps), Math.max(...timestamps)]);
          }
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
          
          if (formattedData.length > 0) {
            const timestamps = formattedData.map(d => d.x);
            setFullDomain([Math.min(...timestamps), Math.max(...timestamps)]);
          }
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
  
  // Toggle scatter plot (raw data points)
  const toggleScatter = async () => {
    if (!rawDataLoaded && !loadingRawData) {
      try {
        setLoadingRawData(true);
        
        const response = await api.get(`/target/${targetName}/raw`, {
          params: { filter_type: filterType, filter_value: filterValue }
        });
        
        const formattedData = response.data.data.map(item => ({
          ...item,
          x: new Date(item.createtime).getTime(),
          Stance: parseFloat(item.Stance) || 0
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
      setShowScatter(!showScatter);
    }
  };
  
  // Handle filter type change
  const handleFilterTypeChange = (e) => {
    const newType = e.target.value;
    // Reset filter value when changing type
    setFilterValue('all');
    setFilterType(newType);
  };
  
  // Handle filter value change
  const handleFilterValueChange = (e) => {
    setFilterValue(e.target.value);
  };
  
  // Formatter for multiple timeline tooltips
  const formatTooltipMultiple = (value, name) => {
    if (name.startsWith('trend_mean_')) {
      const filterVal = name.replace('trend_mean_', '');
      return [value !== undefined ? value.toFixed(2) : 'N/A', `Stance (${filterVal})`];
    }
    if (name.startsWith('trend_lower_')) {
      const filterVal = name.replace('trend_lower_', '');
      return [value !== undefined ? value.toFixed(2) : 'N/A', `Lower CI (${filterVal})`];
    }
    if (name.startsWith('trend_upper_')) {
      const filterVal = name.replace('trend_upper_', '');
      return [value !== undefined ? value.toFixed(2) : 'N/A', `Upper CI (${filterVal})`];
    }
    if (name.startsWith('ci_base_') || name.startsWith('ci_fill_')) {
      // Hide these computed fields from tooltip
      return null;
    }
    if (name.startsWith('volume_')) {
      const filterVal = name.replace('volume_', '');
      return [value !== undefined ? value : 'N/A', `Volume (${filterVal})`];
    }
    return [value, name];
  };

  // Render multiple timelines chart with confidence intervals
  const renderMultipleTimelinesChart = () => {
    
    return (
      <>
        <div className="stance-chart">
          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={trendData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="x" 
                type="number" 
                scale="time"
                domain={visibleDomain}
                tickFormatter={formatAxisDate}
                padding={{ left: 10, right: 10 }}
                allowDataOverflow={true}
              />
              <YAxis 
                domain={[-1, 1]} 
                ticks={[-1, -0.5, 0, 0.5, 1]}
                tickFormatter={(value) => {
                  if (value === -1) return 'Against';
                  if (value === 0) return 'Neutral';
                  if (value === 1) return 'For';
                  return '';
                }}
              />
              <Tooltip 
                labelFormatter={formatTooltipDate}
                formatter={formatTooltipMultiple}
              />
              <Legend />
              <ReferenceLine y={0} stroke="#666" strokeDasharray="3 3" />
              
              {availableFilterValues.map((filterVal, index) => {
                const color = getFilterColor(filterVal, index);
                const areaId = `colorConfidence_${filterVal}`;
                
                return (
                  <React.Fragment key={filterVal}>
                    <defs>
                      <radialGradient id={areaId} cx="50%" cy="50%" r="50%">
                        <stop offset="0%" stopColor={color} stopOpacity={0.8}/>
                        <stop offset="70%" stopColor={color} stopOpacity={0.4}/>
                        <stop offset="100%" stopColor={color} stopOpacity={0.1}/>
                      </radialGradient>
                    </defs>
                    
                    <Area 
                      dataKey={`ci_base_${filterVal}`}
                      stackId={`confidence_${filterVal}`}
                      stroke="none"
                      fill="transparent"
                      name={`CI Base (${filterVal})`}
                      activeDot={false}
                    />
                    
                    <Area 
                      dataKey={`ci_fill_${filterVal}`}
                      stackId={`confidence_${filterVal}`}
                      stroke="none"
                      fill={color}
                      fillOpacity={0.2}
                      name={`CI (${filterVal})`}
                      activeDot={false}
                    />
                    
                    <Line 
                      type="monotone" 
                      dataKey={`trend_lower_${filterVal}`}
                      name={`Lower CI (${filterVal})`}
                      stroke={color}
                      strokeWidth={1}
                      strokeDasharray="5 5"
                      dot={false}
                      activeDot={false}
                      connectNulls={true}
                    />
                    
                    <Line 
                      type="monotone" 
                      dataKey={`trend_upper_${filterVal}`}
                      name={`Upper CI (${filterVal})`}
                      stroke={color}
                      strokeWidth={1}
                      strokeDasharray="5 5"
                      dot={false}
                      activeDot={false}
                      connectNulls={true}
                    />
                    
                    <Line 
                      type="monotone" 
                      dataKey={`trend_mean_${filterVal}`}
                      name={filterVal}
                      stroke={color}
                      strokeWidth={2}
                      dot={false}
                      activeDot={{ r: 6 }}
                      connectNulls={true}
                    />
                  </React.Fragment>
                );
              })}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        
        <div className="volume-chart">
          <ResponsiveContainer width="100%" height={150}>
            <ComposedChart data={trendData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="x" 
                type="number"
                scale="time"
                domain={visibleDomain}
                tickFormatter={formatAxisDate}
                padding={{ left: 10, right: 10 }}
                allowDataOverflow={true}
              />
              <YAxis />
              <Tooltip 
                labelFormatter={formatTooltipDate}
                formatter={formatTooltipMultiple}
              />
              <Legend />
              
              {availableFilterValues.map((filterVal, index) => (
                <Line 
                  key={filterVal}
                  type="monotone" 
                  dataKey={`volume_${filterVal}`}
                  name={filterVal}
                  stroke={getFilterColor(filterVal, index)}
                  connectNulls={true}
                />
              ))}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </>
    );
  };

  // Render single timeline chart
  const renderSingleTimelineChart = () => {
    
    return (
      <>
        <div className="stance-chart">
          <ResponsiveContainer width="100%" height={200}>
            <ComposedChart data={trendData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="x" 
                type="number" 
                scale="time"
                domain={visibleDomain}
                tickFormatter={formatAxisDate}
                padding={{ left: 10, right: 10 }}
                allowDataOverflow={true}
              />
              <YAxis 
                domain={[-1, 1]} 
                ticks={[-1, -0.5, 0, 0.5, 1]}
                tickFormatter={(value) => {
                  if (value === -1) return 'Against';
                  if (value === 0) return 'Neutral';
                  if (value === 1) return 'For';
                  return '';
                }}
              />
              <Tooltip 
                labelFormatter={formatTooltipDate}
                formatter={(value, name) => {
                  if (name === 'trend_mean') return [value.toFixed(2), 'Stance'];
                  if (name === 'trend_lower') return [value.toFixed(2), 'Lower CI'];
                  if (name === 'trend_upper') return [value.toFixed(2), 'Upper CI'];
                  if (name === 'CI Base' || name === 'Confidence Interval') return null;
                  return [value, name];
                }}
              />
              <ReferenceLine y={0} stroke="#666" strokeDasharray="3 3" />
              
              <defs>
                <radialGradient id="colorConfidence" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="#8884d8" stopOpacity={0.8}/>
                  <stop offset="70%" stopColor="#8884d8" stopOpacity={0.4}/>
                  <stop offset="100%" stopColor="#8884d8" stopOpacity={0.1}/>
                </radialGradient>
              </defs>
              
              <Area 
                type="monotone" 
                dataKey="ci_base"
                stackId="confidence"
                stroke="none"
                fill="transparent"
                name="CI Base"
              />
              
              <Area 
                type="monotone" 
                dataKey="ci_fill"
                stackId="confidence"
                stroke="none"
                fillOpacity={0.3}
                fill="#8884d8"
                name="Confidence Interval"
              />
              
              <Line 
                type="monotone" 
                dataKey="trend_lower"
                stroke="#8884d8" 
                strokeWidth={1}
                strokeDasharray="5 5"
                dot={false}
                activeDot={false}
                name="Lower CI"
              />
              
              <Line 
                type="monotone" 
                dataKey="trend_upper"
                stroke="#8884d8" 
                strokeWidth={1}
                strokeDasharray="5 5"
                dot={false}
                activeDot={false}
                name="Upper CI"
              />
              
              <Line 
                type="monotone" 
                dataKey="trend_mean"
                stroke="#8884d8" 
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 6 }}
                name="Average stance"
              />
              
              {showScatter && rawDataLoaded && (
                <Scatter
                  data={rawData}
                  fill="#1E90FF"
                  opacity={0.5}
                  name="Data points"
                  xAxisKey="x"
                  yAxisKey="Stance"
                />
              )}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        
        <div className="volume-chart">
          <ResponsiveContainer width="100%" height={100}>
            <AreaChart data={trendData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="x" 
                type="number"
                scale="time"
                domain={visibleDomain}
                tickFormatter={formatAxisDate}
                padding={{ left: 10, right: 10 }}
                allowDataOverflow={true}
              />
              <YAxis />
              <Tooltip 
                labelFormatter={formatTooltipDate}
                formatter={(value) => [value, 'Volume']}
              />
              
              <defs>
                <linearGradient id="colorVolume" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#82ca9d" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#82ca9d" stopOpacity={0.3}/>
                </linearGradient>
              </defs>
              
              <Area 
                type="monotone" 
                dataKey="volume" 
                stroke="#82ca9d" 
                fill="url(#colorVolume)"
                name="Volume"
              />
            </AreaChart>
          </ResponsiveContainer>
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
    return (
      <div className="target-chart no-data">
        <h2>{targetName}</h2>
        <p>No data available for this target</p>
      </div>
    );
  }

  const isZoomed = zoomLevel > 1;
  const showingMultipleTimelines = filterType !== 'all' && filterValue === 'all';

  return (
    <div className="target-chart">
      <h2>{targetName}</h2>
      
      <div className="chart-controls">
        <div className="filter-controls">
          <div className="filter-group">
            <label>Filter by:</label>
            <select value={filterType} onChange={handleFilterTypeChange}>
              <option value="all">All</option>
              <option value="platform">Platform</option>
              <option value="party">Party</option>
            </select>
          </div>
          
          {filterType !== 'all' && (
            <div className="filter-group">
              <label>Select value:</label>
              <select 
                value={filterValue} 
                onChange={handleFilterValueChange}
                disabled={filterType === 'all'}
              >
                <option value="all">All {filterType}s</option>
                {availableFilterValues.map(value => (
                  <option key={value} value={value}>{value}</option>
                ))}
              </select>
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
          
          {isZoomed && (
            <button 
              type="button" 
              className="chart-button reset-zoom-button"
              onClick={handleResetZoom}
            >
              Reset Zoom
            </button>
          )}
        </div>
      </div>
      
      <div 
        ref={chartContainerRef} 
        className="charts-container"
        onMouseDown={handleMouseDown}
        style={{ cursor: isPanning ? 'grabbing' : (isZoomed ? 'grab' : 'default') }}
      >
        {showingMultipleTimelines ? 
          renderMultipleTimelinesChart() : 
          renderSingleTimelineChart()
        }
      </div>
      
      <div className="chart-instructions">
        <p>
          <strong>Zoom:</strong> Mouse wheel to zoom in/out. {isZoomed && <span><strong>Pan:</strong> Click and drag. <strong>Reset:</strong> Use reset button.</span>}
          {showingMultipleTimelines && <span> <strong>Note:</strong> Select 'All {filterType}s' to compare timelines together.</span>}
        </p>
      </div>
    </div>
  );
};

export default TargetChart;