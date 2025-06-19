import React, { createContext, useState, useEffect, useContext } from 'react';
import { getCurrentUser, isAuthenticated, logoutUser } from '../services/api';

// Create the Auth Context
const AuthContext = createContext(null);

// Create a provider component
export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);
    const [authenticated, setAuthenticated] = useState(false);
    
    // Check if we should skip authentication in development
    const skipAuth = process.env.REACT_APP_SKIP_AUTH === 'true';
  
    useEffect(() => {
      const checkAuthStatus = async () => {
  setLoading(true);
  
  // Skip auth check in development if enabled
  if (skipAuth && process.env.NODE_ENV === 'development') {
    console.log("Development mode: Skipping authentication");
    setAuthenticated(true);
    setUser({ username: 'dev-user', full_name: 'Development User' });
    setLoading(false);
    return;
  }
  
  // Normal authentication flow for production
  const authStatus = isAuthenticated();
  
  if (authStatus) {
    try {
      const response = await getCurrentUser();
      if (response.success) {
        setUser(response.user);
        setAuthenticated(true);
      } else {
        // Don't automatically redirect, just update the auth state
        setUser(null);
        setAuthenticated(false);
      }
    } catch (err) {
      console.error('Error fetching user data:', err);
      // Don't automatically redirect, just update the auth state
      setUser(null);
      setAuthenticated(false);
    }
  } else {
    setUser(null);
    setAuthenticated(false);
  }
  
  setLoading(false);
};
      
      checkAuthStatus();
    }, [skipAuth]);

  // Function to update authentication status after login
  const handleLogin = async () => {
    setLoading(true);
    setAuthenticated(true);
    
    try {
      const response = await getCurrentUser();
      if (response.success) {
        setUser(response.user);
      }
    } catch (err) {
      console.error('Error fetching user data after login:', err);
    } finally {
      setLoading(false);
    }
  };

  // Function to handle logout
  const handleLogout = () => {
    setUser(null);
    setAuthenticated(false);
    logoutUser();
  };

  // Value to be provided to consumers of this context
  const value = {
    user,
    loading,
    authenticated,
    login: handleLogin,
    logout: handleLogout
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

// Custom hook to use the auth context
export const useAuth = () => {
    const context = useContext(AuthContext);
    if (!context) {
      throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
  };

export default AuthContext;