import React from 'react';
import { Navigate, Outlet, useLocation } from 'react-router-dom';
import { useAuth } from './AuthContext';

const PrivateRoute = () => {
  const { authenticated, loading } = useAuth();
  const location = useLocation();
  console.log(process.env.AUTH_URL_PATH);
  const skipAuth = process.env.REACT_APP_SKIP_AUTH === 'true' || process.env.AUTH_URL_PATH === undefined;

  // Get the current path for redirect after login, but avoid including query parameters
  // to prevent potential infinite redirect loops
  const currentPath = location.pathname;
  const loginPath = `/login?returnUrl=${encodeURIComponent(currentPath)}`;

  if (loading) {
    return (
      <div className="loading-container">
        <div className="loading">Checking authentication...</div>
      </div>
    );
  }
  
  // Always allow access in development mode if skip auth is enabled
  if (skipAuth) {
    return <Outlet />;
  }
  
  // Normal authentication check for production
  return authenticated ? <Outlet /> : <Navigate to={loginPath} replace />;
};

export default PrivateRoute;