/**
 * Format a date for display
 * @param {Date|string} date - Date object or date string
 * @param {object} options - Formatting options
 * @returns {string} Formatted date string
 */
export const formatDate = (date, options = {}) => {
    if (!date) return '';
    
    const dateObj = date instanceof Date ? date : new Date(date);
    
    // Default options
    const defaultOptions = {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      ...options
    };
    
    try {
      return dateObj.toLocaleDateString(undefined, defaultOptions);
    } catch (error) {
      console.error('Error formatting date:', error);
      return String(date);
    }
  };
  
  /**
   * Format a number with specified precision
   * @param {number} value - The number to format
   * @param {number} precision - Number of decimal places
   * @returns {string} Formatted number
   */
  export const formatNumber = (value, precision = 2) => {
    if (value === null || value === undefined) return '';
    
    try {
      return Number(value).toFixed(precision);
    } catch (error) {
      console.error('Error formatting number:', error);
      return String(value);
    }
  };
  
  /**
   * Truncate text to a specified length
   * @param {string} text - Text to truncate
   * @param {number} maxLength - Maximum length
   * @param {string} ellipsis - String to indicate truncation
   * @returns {string} Truncated text
   */
  export const truncateText = (text, maxLength = 100, ellipsis = '...') => {
    if (!text) return '';
    
    if (text.length <= maxLength) return text;
    
    return text.slice(0, maxLength) + ellipsis;
  };
  
  /**
   * Format a stance value to a human-readable stance label
   * @param {number} stance - Stance value between -1 and 1
   * @returns {string} Stance label
   */
  export const formatStance = (stance) => {
    if (stance === null || stance === undefined) return 'Unknown';
    
    if (stance <= -0.67) return 'Strongly Against';
    if (stance <= -0.33) return 'Against';
    if (stance <= -0.1) return 'Slightly Against';
    if (stance >= 0.67) return 'Strongly For';
    if (stance >= 0.33) return 'For';
    if (stance >= 0.1) return 'Slightly For';
    
    return 'Neutral';
  };
  
  /**
   * Get a color based on stance value
   * @param {number} stance - Stance value between -1 and 1
   * @returns {string} Color hex code
   */
  export const getStanceColor = (stance) => {
    if (stance === null || stance === undefined) return '#888888';
    
    // Red (against) to Blue (for) gradient
    if (stance <= -0.67) return '#D32F2F'; // Strong red
    if (stance <= -0.33) return '#F44336'; // Red
    if (stance <= -0.1) return '#FFCDD2';  // Light red
    if (stance >= 0.67) return '#1565C0';  // Strong blue
    if (stance >= 0.33) return '#2196F3';  // Blue
    if (stance >= 0.1) return '#BBDEFB';   // Light blue
    
    return '#E0E0E0'; // Neutral gray
  };