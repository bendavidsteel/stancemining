import React, { createContext, useState, useEffect, useContext } from 'react';
import { getCurrentUser, isAuthenticated, logoutUser } from '../services/api';

// Create the Auth Context
const AuthContext = createContext(null);

// Create a provider component
export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);
    const [authenticated, setAuthenticated] = useState(false);
    
    // Skip auth if explicitly disabled or no auth URL configured
    const skipAuth = process.env.REACT_APP_SKIP_AUTH === 'true' || process.env.REACT_APP_AUTH_URL_PATH === undefined;
    
    useEffect(() => {
      const checkAuthStatus = async () => {
  setLoading(true);
  
  if (skipAuth) {
    console.log("Authentication disabled");
    setAuthenticated(true);
    setUser({ username: 'dev-user', full_name: 'Development User' });
    setLoading(false);
    return;
  }
  
  // Normal authentication flow
  const authStatus = isAuthenticated();
  
  if (authStatus) {
    try {
      const response = await getCurrentUser();
      if (response.success) {
        setUser(response.user);
        setAuthenticated(true);
      } else {
        setUser(null);
        setAuthenticated(false);
      }
    } catch (err) {
      console.error('Error fetching user data:', err);
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
    }, []);

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
    authEnabled: !skipAuth,
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