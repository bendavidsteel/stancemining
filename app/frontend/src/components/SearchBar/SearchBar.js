import React, { useState, useEffect } from 'react';
import './SearchBar.css';

const SearchBar = ({ onSearch, onClear, initialValue = '' }) => {
  const [query, setQuery] = useState(initialValue);

  useEffect(() => {
    setQuery(initialValue);
  }, [initialValue]);

  const handleInputChange = (e) => {
    setQuery(e.target.value);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSearch(query);
  };

  const handleClear = () => {
    setQuery('');
    onClear();
  };

  return (
    <div className="search-bar">
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={query}
          onChange={handleInputChange}
          placeholder="Search stance targets..."
          className="search-input"
        />
        <button 
          type="submit" 
          className="search-button"
          disabled={!query.trim()}
        >
          Search
        </button>
        <button 
          type="button" 
          className="clear-button"
          onClick={handleClear}
        >
          Clear
        </button>
      </form>
    </div>
  );
};

export default SearchBar;