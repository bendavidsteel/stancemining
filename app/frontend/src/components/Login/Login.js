import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { loginUser } from '../../services/api';
import { useAuth } from '../../contexts/AuthContext';
import './Login.css';

const Login = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const { login } = useAuth(); // Get login function from auth context
  
  // Extract return URL from query parameters
  const searchParams = new URLSearchParams(location.search);
  const returnUrl = searchParams.get('returnUrl') || '/';
  
  // Prevent redirect loops - if returnUrl points to login, redirect to home
  const safeReturnUrl = returnUrl.startsWith('/login') ? '/' : returnUrl;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    // Simple validation
    if (!username.trim() || !password.trim()) {
      setError('Username and password are required');
      setLoading(false);
      return;
    }

    try {
      const result = await loginUser(username, password);
      
      if (result.success) {
        // Update auth context
        await login();
        // Redirect to return URL or dashboard on successful login
        navigate(safeReturnUrl);
      } else {
        setError(result.message || 'Invalid username or password');
      }
    } catch (err) {
      setError('Login failed. Please try again.');
      console.error('Login error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-container">
      <div className="login-form-container">
        <h1>Stance Target Dashboard</h1>
        <h2>Login</h2>
        
        {error && <div className="error-message">{error}</div>}
        
        <form onSubmit={handleSubmit} className="login-form">
          <div className="form-group">
            <label htmlFor="username">Username</label>
            <input
              type="text"
              id="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter your username"
              disabled={loading}
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter your password"
              disabled={loading}
            />
          </div>
          
          <button 
            type="submit" 
            className="login-button"
            disabled={loading}
          >
            {loading ? 'Logging in...' : 'Login'}
          </button>
        </form>
      </div>
    </div>
  );
};

export default Login;