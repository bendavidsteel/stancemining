# Stance Dashboard Web Application

This repository contains a web application with a separate frontend and backend architecture.

## Project Structure

The project consists of two main parts:

### 1. Backend API (FastAPI)

Located in the `/backend` directory:

- `main.py`: The main FastAPI application that serves as the backend API
- Data handling using Polars for efficient DataFrame operations
- Endpoints for retrieving stance data, filtering, and semantic search

### 2. Frontend Application (React)

Located in the `/frontend` directory:

- React-based single-page application
- Interactive charts using Recharts
- Components for search, pagination, and data visualization

## Setup Instructions

### Backend Setup

1. Create a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install fastapi uvicorn polars numpy sentence-transformers
   ```

3. Start the backend server:
   ```bash
   uvicorn main:app --reload
   ```

The backend will run on http://localhost:8000 by default.

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

The frontend will run on http://localhost:3000 by default.

## API Documentation

Once the backend is running, you can access the auto-generated API documentation at:
- http://localhost:8000/docs (Swagger UI)
- http://localhost:8000/redoc (ReDoc)

## Data Directory Structure

Make sure your data is organized as follows:

```
/data
  /precomputed
    - valid_targets_list.parquet.zstd
    - all_targets_trends.parquet.zstd
    - all_targets_raw.parquet.zstd
    - target_embeddings.parquet.zstd
    - platforms.parquet.zstd
    - parties.parquet.zstd
```

## Features

- **Paginated Targets**: Browse through all available stance targets
- **Semantic Search**: Find relevant targets using natural language search
- **Interactive Charts**: Visualize stance trends and volume data
- **Filtering**: Filter data by platform or party
- **Raw Data Display**: Option to show individual data points