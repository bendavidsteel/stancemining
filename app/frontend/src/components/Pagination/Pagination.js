import React from 'react';
import './Pagination.css';

const Pagination = ({ currentPage, totalPages, onPageChange }) => {
  const handlePrevious = () => {
    if (currentPage > 0) {
      onPageChange(currentPage - 1);
    }
  };

  const handleNext = () => {
    if (currentPage < totalPages - 1) {
      onPageChange(currentPage + 1);
    }
  };

  return (
    <div className="pagination">
      <button
        className="pagination-button"
        onClick={handlePrevious}
        disabled={currentPage === 0}
      >
        Previous
      </button>
      <div className="page-info">
        Page {currentPage + 1} of {totalPages}
      </div>
      <button
        className="pagination-button"
        onClick={handleNext}
        disabled={currentPage === totalPages - 1}
      >
        Next
      </button>
    </div>
  );
};

export default Pagination;