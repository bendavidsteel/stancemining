import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation, Link, useNavigate } from 'react-router-dom';
import SearchBar from './components/SearchBar';
import Pagination from './components/Pagination';
import TargetChart from './components/TargetChart';
import UmapVisualization from './components/UmapVisualization';
import Login from './components/Login';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import PrivateRoute from './contexts/PrivateRoute';
import { getTargets, searchTargets, API_BASE_URL, isAuthenticated } from './services/api';
import './App.css';

// Navigation component
const Navbar = ({ onLogout }) => {
  const location = useLocation();
  const { authEnabled } = useAuth();
  
  return (
    <nav className="App-navbar">
      <div className="navbar-logo">Stance Target Dashboard</div>
      <div className="navbar-tabs">
        <Link 
          to="/" 
          className={`navbar-tab ${location.pathname === '/' ? 'active' : ''}`}
        >
          Timeline View
        </Link>
        <Link 
          to="/map" 
          className={`navbar-tab ${location.pathname === '/map' ? 'active' : ''}`}
        >
          Map View
        </Link>
      </div>
      <div className="navbar-actions">
        {authEnabled && (
          <button onClick={onLogout} className="logout-button">
            Logout
          </button>
        )}
      </div>
    </nav>
  );
};

// Main dashboard component for timeline view
const TimelineView = () => {
  const { logout } = useAuth();
  const location = useLocation();
  const [targets, setTargets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [currentPage, setCurrentPage] = useState(0);
  const [totalPages, setTotalPages] = useState(0);
  const [searchMode, setSearchMode] = useState(false);
  const [searchResults, setSearchResults] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTarget, setSelectedTarget] = useState(null);

  // Check for target parameter in URL for direct linking
  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const targetParam = params.get('target');
    if (targetParam) {
      setSelectedTarget(targetParam);
      setSearchMode(true);
      setSearchResults([{ Target: targetParam }]);
      setCurrentPage(0);
      setTotalPages(1);
    }
  }, [location.search]);

  useEffect(() => {
    if (!selectedTarget) {
      fetchTargets(currentPage);
    }
  }, [currentPage, selectedTarget]);

  const fetchTargets = async (page) => {
    try {
      setLoading(true);
      const response = await getTargets(page, 5);

      setTargets(response.targets);
      setTotalPages(response.total_pages);
      setLoading(false);
    } catch (err) {
      console.error('Error fetching targets:', err);
      setError('Failed to load targets. Please try again later.');
      setLoading(false);
    }
  };

  const handleSearch = async (query) => {
    if (!query.trim()) {
      clearSearch();
      return;
    }

    try {
      setLoading(true);
      setSearchQuery(query);
      const response = await searchTargets(query);

      setSearchResults(response.results);
      setSearchMode(true);
      setCurrentPage(0);
      setTotalPages(Math.ceil(response.results.length / 5));
      setLoading(false);
    } catch (err) {
      console.error('Error searching targets:', err);
      setError('Search failed. Please try again.');
      setLoading(false);
    }
  };

  const clearSearch = () => {
    setSearchMode(false);
    setSearchQuery('');
    setSearchResults([]);
    setSelectedTarget(null);
    setCurrentPage(0);
    fetchTargets(0);
  };

  const handlePageChange = (newPage) => {
    setCurrentPage(newPage);
  };

  // Get current targets to display based on pagination and search mode
  const getCurrentTargets = () => {
    if (selectedTarget) {
      return searchResults;
    }
    
    const itemsPerPage = 5;
    const startIndex = currentPage * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;

    if (searchMode) {
      return searchResults.slice(startIndex, endIndex);
    } else {
      return targets;
    }
  };

  return (
    <div className="view-container">
      <SearchBar
        onSearch={handleSearch}
        onClear={clearSearch}
        initialValue={searchQuery}
      />

      {searchMode && !selectedTarget && (
        <div className="search-results-info">
          Found {searchResults.length} targets matching '{searchQuery}'
        </div>
      )}

      {!selectedTarget && (
        <Pagination
          currentPage={currentPage}
          totalPages={totalPages}
          onPageChange={handlePageChange}
        />
      )}

      {loading ? (
        <div className="loading">Loading...</div>
      ) : error ? (
        <div className="error">{error}</div>
      ) : getCurrentTargets().length === 0 ? (
        <div className="no-results">No targets to display</div>
      ) : (
        <div className="target-charts">
          {getCurrentTargets().map((target) => (
            <TargetChart
              key={target.Target}
              targetName={target.Target}
              apiBaseUrl={API_BASE_URL}
            />
          ))}
        </div>
      )}

      {!loading && !error && getCurrentTargets().length > 0 && !selectedTarget && (
        <Pagination
          currentPage={currentPage}
          totalPages={totalPages}
          onPageChange={handlePageChange}
        />
      )}
    </div>
  );
};

// Map view component
const MapView = () => {
  return (
    <div className="view-container">
      <UmapVisualization />
    </div>
  );
};

// Main layout component with navigation
const MainLayout = () => {
  const { logout } = useAuth();
  
  return (
    <div className="App">
      <Navbar onLogout={logout} />
      <main className="App-main">
        <Routes>
          <Route path="/" element={<TimelineView />} />
          <Route path="/map" element={<MapView />} />
        </Routes>
      </main>
    </div>
  );
};

// Login route with redirect
const LoginRoute = () => {
  const { authenticated } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  
  // Extract return URL from query parameters
  const searchParams = new URLSearchParams(location.search);
  const returnUrl = searchParams.get('returnUrl') || '/';
  
  // Prevent redirect loops - if returnUrl points to login, redirect to home
  const safeReturnUrl = returnUrl.startsWith('/login') ? '/' : returnUrl;
  
  // If already authenticated, redirect to the return URL
  useEffect(() => {
    if (authenticated) {
      navigate(safeReturnUrl, { replace: true });
    }
  }, [authenticated, navigate, safeReturnUrl]);
  
  return <Login />;
};

// Main App component with routing
function App() {
  return (
    <AuthProvider>
      <Router>
        <Routes>
          {/* Public routes */}
          <Route path="/login" element={<LoginRoute />} />
          
          {/* Protected routes */}
          <Route element={<PrivateRoute />}>
            <Route path="/*" element={<MainLayout />} />
          </Route>
          
          {/* Redirect any other routes to dashboard if authenticated, or login if not */}
          <Route path="*" element={<Navigate to="/" />} />
        </Routes>
      </Router>
    </AuthProvider>
  );
}

export default App;