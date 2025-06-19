import axios from 'axios';

export const API_BASE_URL = 'http://localhost:8000';

// Helper function to set a cookie
const setCookie = (name, value, days = 1) => {
  const expires = new Date();
  expires.setTime(expires.getTime() + days * 24 * 60 * 60 * 1000);
  document.cookie = `${name}=${value};expires=${expires.toUTCString()};path=/`;
};

// Helper function to get a cookie
const getCookie = (name) => {
  const value = `; ${document.cookie}`;
  const parts = value.split(`; ${name}=`);
  if (parts.length === 2) return parts.pop().split(';').shift();
  return null;
};

// Helper function to delete a cookie
const deleteCookie = (name) => {
  document.cookie = `${name}=;expires=Thu, 01 Jan 1970 00:00:00 GMT;path=/`;
};

// Create an axios instance with auth token interceptor
const api = axios.create({
  baseURL: API_BASE_URL
});

// Add a request interceptor to include auth token from cookie
api.interceptors.request.use(
  (config) => {
    const token = getCookie('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Add a response interceptor to handle auth errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      // If unauthorized, clear token but don't redirect automatically
      // This prevents potential redirect loops
      deleteCookie('authToken');
      
      // If we're not already on the login page, we can redirect
      if (!window.location.pathname.startsWith('/login')) {
        // Get current path to redirect back after login
        const currentPath = window.location.pathname + window.location.search;
        const returnUrl = `?returnUrl=${encodeURIComponent(currentPath)}`;
        
        // Redirect to login page
        window.location.href = `/login${returnUrl}`;
      }
    }
    return Promise.reject(error);
  }
);

// Login function - uses the backend as a proxy to the external API
export const loginUser = async (username, password) => {
  try {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);
    
    const response = await axios.post(`${API_BASE_URL}/token`, formData);
    
    if (response.data.access_token) {
      // Store in cookie
      setCookie('authToken', response.data.access_token);
      return { success: true };
    }
    return { success: false, message: 'Invalid credentials' };
  } catch (error) {
    console.error('Login error:', error);
    return { 
      success: false, 
      message: error.response?.data?.detail || 'Login failed' 
    };
  }
};

// Update logout function
export const logoutUser = () => {
  // Remove from cookie
  deleteCookie('authToken');
  
  // Only redirect if not already on login page
  if (!window.location.pathname.startsWith('/login')) {
    window.location.href = '/login';
  }
  
  // Return true to indicate successful logout
  return true;
};

export const getCurrentUser = async () => {
  try {
    const response = await api.get('/users/me');
    return { success: true, user: response.data };
  } catch (error) {
    return { success: false, message: 'Failed to get user info' };
  }
};

// Update authentication check
export const isAuthenticated = () => {
  return getCookie('authToken') !== null;
};

// API functions - now using the authenticated api instance
export const getTargets = async (page = 0, perPage = 5) => {
  const response = await api.get('/targets', { params: { page, per_page: perPage } });
  return response.data;
};

export const searchTargets = async (query) => {
  const response = await api.get('/search', { params: { query } });
  return response.data;
};

export const getTargetTrends = async (targetName, filterType = 'all', filterValue = 'all') => {
  const response = await api.get(`/target/${targetName}/trends`, {
    params: { filter_type: filterType, filter_value: filterValue }
  });
  return response.data;
};

export const getTargetRawData = async (targetName, filterType = 'all', filterValue = 'all') => {
  const response = await api.get(`/target/${targetName}/raw`, {
    params: { filter_type: filterType, filter_value: filterValue }
  });
  return response.data;
};

export const getTargetFilters = async (targetName) => {
  const response = await api.get(`/target/${targetName}/filters`);
  return response.data;
};

export const getAllFilters = async () => {
  const response = await api.get('/filters');
  return response.data;
};

// New API function for UMAP visualization data
export const getUmapData = async () => {
  const response = await api.get('/umap');
  return response.data;
};

// Error handling utility
export const handleApiError = (error) => {
  let errorMessage = 'An error occurred. Please try again.';
  
  if (error.response) {
    // The request was made and the server responded with a status code
    // that falls out of the range of 2xx
    console.error('Response error:', error.response.data);
    errorMessage = error.response.data.detail || errorMessage;
  } else if (error.request) {
    // The request was made but no response was received
    console.error('Request error:', error.request);
    errorMessage = 'No response from server. Please check your connection.';
  } else {
    // Something happened in setting up the request that triggered an Error
    console.error('Error:', error.message);
    errorMessage = error.message || errorMessage;
  }
  
  return errorMessage;
};

export default api;