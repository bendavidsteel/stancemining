from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import os
import random
import re
import requests

import numpy as np
import polars as pl
from pydantic import BaseModel
import pynndescent
from sentence_transformers import SentenceTransformer
from jose import jwt, JWTError
import sklearn.metrics.pairwise

from fastapi import FastAPI, Query, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.security import OAuth2PasswordRequestForm


# Set a fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

app = FastAPI(title="Stance Dashboard API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict this in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
    expose_headers=["Authorization"]
)

# Load environment variables
EXTERNAL_API_BASE_URL = os.getenv("API_BASE_URL", "https://api.meoinsightshub.net")
API_USERNAME = os.getenv("MEO_USERNAME")
API_PASSWORD = os.getenv("MEO_PASSWORD")
DATA_DIR_PATH = os.getenv('DATA_DIR_PATH')

# Bearer token security
security = HTTPBearer()

# --- Authentication Models ---
class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    
class Token(BaseModel):
    access_token: str
    token_type: str
    
class TokenData(BaseModel):
    username: Optional[str] = None

# --- Original Models ---
class Target(BaseModel):
    Target: str
    count: int
    
    class Config:
        # Allow extra fields that might be in the data but not in the model
        extra = "ignore"

class TrendPoint(BaseModel):
    createtime: str
    trend_mean: float
    trend_lower: float
    trend_upper: float
    volume: int
    filter_type: str
    filter_value: str

class RawDataPoint(BaseModel):
    Target: str
    SeedName: str
    document_text: str
    Stance: float
    createtime: str
    platform: str
    Party: str

class SearchResponse(BaseModel):
    results: List[Target]
    total: int

class TargetResponse(BaseModel):
    targets: List[Target]
    total: int
    page: int
    total_pages: int

class UmapPoint(BaseModel):
    Target: str
    x: float
    y: float
    count: Optional[int] = None
    avg_stance: Optional[float] = None
    stance_std: Optional[float] = None
    stance_abs: Optional[float] = None
    n_positive: Optional[int] = None
    n_negative: Optional[int] = None
    n_neutral: Optional[int] = None
    top_platform: Optional[str] = None
    top_party: Optional[str] = None
    top_main_type: Optional[str] = None
    secondary_targets: Optional[str] = None

class UmapResponse(BaseModel):
    data: List[UmapPoint]

# Global variables to store loaded data
target_df = None
all_trends_df = None
target_embeddings_df = None
unique_platforms = []
unique_parties = []
embedding_model = None
nn_index = None

# --- Authentication Helper Functions ---
def get_token(username: str, password: str):
    """Get authentication token from the external API"""
    try:
        res = requests.get(
            f"{EXTERNAL_API_BASE_URL}/meologin", 
            params={"username": username, "password": password}, 
            verify=True
        )
        
        if res.status_code != 200:
            return None
            
        token = res.json().get("access_token")
        return token
    except Exception as e:
        print(f"Error getting token: {e}")
        return None

def verify_token(token: str):
    """Verify token from the external API (placeholder for actual validation)"""
    try:
        # Since we're relying on the external API for validation,
        # we would ideally make a test call to verify the token is still valid
        # For now, we're accepting any non-empty string as valid
        # In production, you would want to validate this properly with the external API
        if token and isinstance(token, str):
            return True
        return False
    except Exception as e:
        print(f"Error verifying token: {e}")
        return False
    

def process_data():
    """Load and process the main data from files using efficient Polars operations"""
    print("Loading and processing data files...")
    # Load data with fewer columns
    dir_path = './data/'
    
    # Collect all file paths first, then read them in a single operation
    file_paths = [
        os.path.join(dir_path, file) 
        for file in os.listdir(dir_path)
        if re.search(r'\d{4}_\d{1,2}_doc_targets_with_stance.parquet.zstd', file)
    ]
    
    if not file_paths:
        raise ValueError("No stance data files found in the data directory")
    
    # Use a scan to potentially improve memory usage
    dfs = [pl.read_parquet(file_path) for file_path in file_paths]
    
    # Concatenate the scanned DataFrames
    df = pl.concat(dfs, how='diagonal_relaxed')
    
    # Filter for consistent list lengths, then explode
    df = (df
          .filter(pl.col('Targets').list.len() == pl.col('Stances').list.len())
          .explode(['Targets', 'Stances'])
          .rename({'Targets': 'Target', 'Stances': 'Stance'})
    )

    # Get ordered list of all targets in a single pipeline
    target_count_df = df.group_by('Target').agg(pl.count().alias('count')).sort('count', descending=True)
    
    # Get unique filter values in a single pipeline
    filter_cols = os.environ['FILTER_COLUMNS'].split(',')
    unique_values = (df
        .select(filter_cols)
        .unique()
    )
    # Save all processed data
    os.makedirs('./data/precomputed', exist_ok=True)
    
    # Use optimized write options
    write_options = {"compression": "zstd", "compression_level": 3}
    
    df.write_parquet('./data/precomputed/processed_stance_data.parquet.zstd', **write_options)
    target_count_df.write_parquet('./data/precomputed/targets_list.parquet.zstd', **write_options)
    
    print(f"Processed {len(df)} records for {len(target_count_df)} targets")
    return df, target_count_df


def compute_target_embeddings(target_count_df: pl.DataFrame):
    """Compute embeddings for targets efficiently"""
    print("Loading sentence transformer model...")
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cuda')
    
    embeddings_dir = './data/precomputed'
    embeddings_path = f'{embeddings_dir}/target_embeddings.parquet.zstd'
    
    # Initialize embeddings dataframe
    embeddings_df = None
    
    # Load cached embeddings if they exist
    if os.path.exists(embeddings_path):
        print(f"Loading cached embeddings from {embeddings_path}")
        embeddings_df = pl.read_parquet(embeddings_path)
        print(f"Loaded {len(embeddings_df)} cached embeddings")
    else:
        # Create empty dataframe if no cache exists
        embeddings_df = pl.DataFrame(schema={'Target': pl.Utf8, 'Embedding': pl.Array(pl.Float32, 384)})
    
    # Using Polars to find missing targets
    if len(embeddings_df) > 0:
        # Create a temporary dataframe with just the targets from target_count_df
        targets_df = target_count_df.select('Target')
        
        # Anti-join to find targets that are not in the embeddings_df
        missing_targets_df = targets_df.join(
            embeddings_df.select('Target'),
            on='Target',
            how='anti'
        )
        
        missing_targets = missing_targets_df['Target'].to_list()
    else:
        missing_targets = target_count_df['Target'].to_list()
    
    # If we have targets that need embedding
    if missing_targets:
        print(f"Computing embeddings for {len(missing_targets)} new targets...")
        new_embeddings = model.encode(missing_targets, show_progress_bar=True)
        
        # Create dataframe for new embeddings
        new_embeddings_df = pl.DataFrame({
            'Target': missing_targets,
            'Embedding': [embedding.tolist() for embedding in new_embeddings]
        }, schema_overrides={'Embedding': pl.Array(pl.Float32, 384)})
        
        # Combine with existing embeddings
        embeddings_df = pl.concat([embeddings_df, new_embeddings_df])
        
        # Save the complete embeddings
        os.makedirs(embeddings_dir, exist_ok=True)
        embeddings_df.write_parquet(
            embeddings_path,
            compression="zstd"
        )
        print(f"Updated target embeddings saved to {embeddings_path}")
    else:
        print("All targets already have embeddings")
    
    # Join embeddings back to the target_count_df
    result_df = target_count_df.join(
        embeddings_df,
        on="Target",
        how="left"
    )
    
    return model, result_df

def calculate_stance_statistics(df, valid_targets: pl.DataFrame):
    """Calculate average stance statistics for each target using efficient Polars operations"""
    target_stats = []
    
    for target_info in valid_targets.to_dicts():
        target_name = target_info['Target']
        
        # Filter data and calculate all statistics in one pipeline
        target_df = df.filter(pl.col('Target') == target_name)
        
        if len(target_df) > 0:
            # Calculate most metrics in a single aggregation
            stats = (target_df
                .select([
                    pl.col('Stance').mean().alias('avg_stance'),
                    pl.col('Stance').std().alias('stance_std'),
                    pl.col('Stance').abs().mean().alias('stance_abs'),
                    pl.when(pl.col('Stance') > 0.1).then(1).otherwise(0).sum().alias('n_positive'),
                    pl.when(pl.col('Stance') < -0.1).then(1).otherwise(0).sum().alias('n_negative'),
                    pl.count().alias('total_count')
                ])
                .with_columns(
                    (pl.col('total_count') - pl.col('n_positive') - pl.col('n_negative')).alias('n_neutral')
                )
                .to_dicts()[0]
            )
            
            # Get top values for categorical fields in one operation
            top_values = {}
            for field, name in [('platform', 'top_platform'), ('Party', 'top_party'), ('MainType', 'top_main_type')]:
                # Get the most common value
                top = target_df.group_by(field).count().sort('count', descending=True).head(1)
                if len(top) > 0:
                    top_values[name] = top[0, field]
                else:
                    top_values[name] = "Unknown"
            
            # Combine all metrics
            target_stat = {
                'Target': target_name,
                'count': target_info.get('count', 0),
                'avg_stance': float(stats['avg_stance']),
                'stance_std': float(stats['stance_std']),
                'stance_abs': float(stats['stance_abs']),
                'n_positive': int(stats['n_positive']),
                'n_negative': int(stats['n_negative']),
                'n_neutral': int(stats['n_neutral']),
                'top_platform': top_values['top_platform'],
                'top_party': top_values['top_party'],
                'top_main_type': top_values['top_main_type'],
            }
            
            target_stats.append(target_stat)
    
    # Create dataframe and save in a single operation
    target_stats_df = pl.DataFrame(target_stats)
    target_stats_df.write_parquet('./data/precomputed/target_statistics.parquet.zstd', compression="zstd")
    print(f"Saved stance statistics for {len(target_stats)} targets")
    
    return target_stats_df

def compute_umap_embeddings(valid_targets_df, target_embeddings_df: pl.DataFrame):
    """Compute UMAP embeddings for visualization with optimized memory usage"""
    print("Computing UMAP embeddings...")
    
    # Extract target names and get corresponding embeddings
    valid_embeddings_df = valid_targets_df.join(target_embeddings_df, on='Target')
    target_names = valid_embeddings_df['Target'].to_list()
    embeddings = valid_embeddings_df['Embedding'].to_numpy()
    
    # Set up UMAP with parameters suitable for visualization
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    
    # Fit UMAP and transform the embeddings
    umap_embeddings = reducer.fit_transform(embeddings)
    
    # Create dataframe with results in a single operation
    umap_df = pl.DataFrame({
        'Target': target_names,
        'x': umap_embeddings[:, 0],
        'y': umap_embeddings[:, 1]
    })
    
    # Optimize the join operations
    cols_to_join = ['Target', 'count']
    umap_df = umap_df.join(valid_targets_df.select(cols_to_join), on='Target', how='left')
    
    # Add stance statistics in a single join operation
    try:
        stats_df = calculate_stance_statistics(df, valid_targets_df)
        stats_cols = [col for col in stats_df.columns if col not in ['Target', 'count', 'secondary_targets']]
        
        umap_df = umap_df.join(
            stats_df.select(['Target'] + stats_cols),
            on='Target', how='left'
        )
    except Exception as e:
        print(f"Warning: Could not add stance statistics to UMAP data: {e}")
    
    # Save UMAP embeddings
    umap_df.write_parquet('./data/precomputed/umap_embeddings.parquet.zstd', compression="zstd")
    print(f"UMAP embeddings saved to ./data/precomputed/umap_embeddings.parquet.zstd")


async def authenticate_request(request: Request):
    """Helper function to authenticate a request with flexibility for development mode"""
    # Handle authentication (with development mode flexibility)
    try:
        # Check if we're in development mode with authentication skipping enabled
        skip_auth = os.getenv("REACT_APP_SKIP_AUTH") == "true" and os.getenv("NODE_ENV") == "development"
        if skip_auth:
            # Skip authentication in development mode if configured
            return User(username="dev-user")
            
        # For other environments, just accept any non-empty token for now
        # In a real implementation, you'd validate against the external API
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return User(username="authenticated_user")
            
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"}
        )

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get the current user from the token"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    if not verify_token(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Since we don't decode the token, we'll create a minimal user object
    # In a real implementation, you might want to decode the token or make an API call
    # to get user details
    return User(username="authenticated_user")

# Alternative option to get current user with optional authentication for development
async def get_current_user_flexible(request: Request):
    """More flexible authentication handler that works with development mode"""
    # Check if we're in development mode with auth skipping enabled
    if os.getenv("NODE_ENV") == "development" and os.getenv("REACT_APP_SKIP_AUTH") == "true":
        return User(username="dev-user")
    
    # Get the Authorization header
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Extract the token
    scheme, token = auth_header.split()
    if scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not verify_token(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return User(username="authenticated_user")

# --- Authentication Routes ---
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login endpoint that forwards authentication to the external API"""
    token = get_token(form_data.username, form_data.password)
    
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return {"access_token": token, "token_type": "bearer"}

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    # In a real implementation, you might want to make an API call to get user details
    return User(
        username=current_user.username,
        full_name="Authenticated User",
    )

@app.on_event("startup")
async def startup_event():
    """Load all precomputed data on startup"""
    global target_df, all_trends_df, target_embeddings_df, unique_platforms, unique_parties, unique_main_types, embedding_model
    
    print("Loading precomputed data...")
    
    try:
        # Load filtered targets list
        target_df = pl.read_parquet(os.path.join(DATA_DIR_PATH, 'precomputed/valid_targets_list.parquet.zstd'))
        
        # Check if 'len' field exists and rename it to 'count' to match our model
        if 'len' in target_df.columns and 'count' not in target_df.columns:
            target_df = target_df.rename({'len': 'count'})

        target_df = target_df.sort('count', descending=True)
        
        print(f"Loaded {len(target_df)} valid targets with ≥50 data points")
        
        # Load trends data
        all_trends_df = pl.read_parquet(os.path.join(DATA_DIR_PATH, 'precomputed', 'all_targets_trends.parquet.zstd'))
        print(f"Loaded unified trends dataframe with {len(all_trends_df)} rows")
        
        # Note: Raw data will be loaded on-demand, not at startup
        
        # Load embeddings for semantic search
        target_embeddings_df = compute_target_embeddings(target_df)
        print(f"Loaded embeddings for {len(target_embeddings_df)} targets")

        # Convert dictionary to a list of embeddings in the same order as all_targets
        embeddings_list = target_df.join(target_embeddings_df, on='Target')['Embedding'].to_numpy()
        
        try:
            # Create the NNDescent index
            global nn_index
            nn_index = pynndescent.NNDescent(embeddings_list)
            print("Initialized PyNNDescent index for fast similarity search")
        except Exception as ex:
            print(f"PyNNDescent failed with error: {ex}")
            
        # Load unique platforms, parties, and main types
        platforms_df = pl.read_parquet(os.path.join(DATA_DIR_PATH, 'precomputed', 'platforms.parquet.zstd'))
        parties_df = pl.read_parquet(os.path.join(DATA_DIR_PATH, 'precomputed', 'parties.parquet.zstd'))
        
        unique_platforms = platforms_df['platform'].to_list()
        unique_parties = parties_df['party'].to_list()
        
        # Load main types if available
        try:
            main_types_df = pl.read_parquet(os.path.join(DATA_DIR_PATH, 'precomputed', 'main_types.parquet.zstd'))
            unique_main_types = main_types_df['main_type'].to_list()
            print(f"Loaded {len(unique_main_types)} main types")
        except Exception as e:
            print(f"Warning: Could not load main types: {e}")
            unique_main_types = []
        
        print(f"Loaded {len(unique_platforms)} platforms and {len(unique_parties)} parties")
        
        # Initialize embedding model
        embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        print("Initialized semantic search model")

        umap_df = compute_umap_embeddings(target_df, target_embeddings_df)
    
    except Exception as e:
        print(f"Error during startup: {e}")
        # Re-raise to prevent the app from starting with incomplete data
        raise e

@app.get("/umap", response_model=UmapResponse)
async def get_umap_data(request: Request):
    """Get UMAP visualization data (requires authentication)"""
    # Authenticate the request
    await authenticate_request(request)
    try:
        # Load UMAP embedding data
        umap_df = pl.read_parquet(os.path.join(DATA_DIR_PATH, 'precomputed', 'umap_embeddings.parquet.zstd'))
        
        # Convert to list of dicts for JSON response
        umap_data = umap_df.to_dicts()
        
        return {"data": umap_data}
    except Exception as e:
        print(f"Error loading UMAP data: {e}")
        return {"data": []}

# --- Protected Routes (require authentication) ---
@app.get("/targets", response_model=TargetResponse)
async def get_targets(
    request: Request,
    page: int = 0, 
    per_page: int = 5
):
    """Get a paginated list of targets (requires authentication)"""
    # Authenticate the request
    await authenticate_request(request)
    start_idx = page * per_page
    end_idx = min(start_idx + per_page, len(target_df))
    
    total_pages = (len(target_df) + per_page - 1) // per_page
    
    return {
        "targets": target_df.slice(start_idx, per_page).to_dicts(),
        "total": len(target_df),
        "page": page,
        "total_pages": total_pages
    }

@app.get("/search", response_model=SearchResponse)
async def search_targets(
    request: Request,
    query: str, 
    top_k: Optional[int] = 20
):
    """Search for targets using semantic search (requires authentication)"""
    # Authenticate the request
    await authenticate_request(request)
    if not query.strip():
        return {"results": [], "total": 0}
    
    if embedding_model and nn_index:
        # Encode the query
        query_embedding = embedding_model.encode([query])[0].reshape(1, -1)
        
        # Use PyNNDescent for efficient nearest neighbor search
        # Get more results than needed to filter by threshold later
        k = min(len(target_df), top_k * 3 if top_k else 100)
        indices, distances = nn_index.query(query_embedding, k=k)
        
        # Convert cosine distance to similarity (1 - distance)
        similarities = 1 - distances[0]
        
        # Filter results by similarity threshold and prepare response
        results = []
        for idx, similarity in zip(indices[0], similarities):
            if similarity > 0.2:  # Keep the same threshold as before
                results.append(target_df['Target'][idx])
                
                # Limit to top_k if specified
                if top_k and len(results) >= top_k:
                    break
    elif embedding_model and nn_index is None:
        query_embedding = embedding_model.encode([query])[0].reshape(1, -1)
        
        # Calculate similarities
        target_embeddings = target_df.join(target_embeddings_df, on='Target', maintain_order='left')['Embedding'].to_numpy()

        cosine_similarity = sklearn.metrics.pairwise.cosine_similarity(query_embedding, target_embeddings).squeeze(0)
        target_similarity_df = target_df.with_columns(pl.Series(name='similarity', values=cosine_similarity))
        target_similarity_df = target_similarity_df.sort('similarity', descending=True)

        if top_k:
            target_similarity_df = target_similarity_df.head(top_k)

        results = target_similarity_df.filter(pl.col('similarity') > 0.2).select(['Target', 'count']).to_dicts()
    else:
        # Fallback to text search
        query = query.lower()
        results = target_df.filter(pl.col('Target').str.to_lowercase().str.contains(query)).select(['Target', 'count']).to_dicts()
    
    return {"results": results, "total": len(results)}

@app.get("/target/{target_name}/trends")
async def get_target_trends(
    request: Request,
    target_name: str, 
    filter_type: str = "all", 
    filter_value: str = "all"
):
    """Get trend data for a specific target with optional filtering (requires authentication)"""
    # Authenticate the request
    await authenticate_request(request)
    # Filter precomputed data for this target
    target_trends = all_trends_df.filter(pl.col('target') == target_name)
    
    if len(target_trends) == 0:
        return {"data": []}
    
    # Apply additional filtering
    if filter_type == "all" or filter_value == "all":
        filtered_trends = target_trends.filter(
            (pl.col('filter_type') == 'all') & (pl.col('filter_value') == 'all')
        )
    else:
        filtered_trends = target_trends.filter(
            (pl.col('filter_type') == filter_type) & (pl.col('filter_value') == filter_value)
        )

    filtered_trends = filtered_trends.drop_nans()
    
    # Convert to list of dicts for JSON response
    return {"data": filtered_trends.to_dicts()}

@app.get("/target/{target_name}/raw")
async def get_target_raw_data(
    request: Request,
    target_name: str, 
    filter_type: str = "all", 
    filter_value: str = "all"
):
    """Get raw data for a specific target with optional filtering (requires authentication)"""
    # Authenticate the request
    await authenticate_request(request)
    try:
        # Load raw data for just this target on-demand
        print(f"Loading raw data for target: {target_name}")
        
        # Load the full raw data file
        # In a production environment, you might want to store individual target files 
        # or use a database that allows querying specific targets efficiently
        raw_df = pl.read_parquet(os.path.join(DATA_DIR_PATH, 'precomputed', 'all_targets_raw.parquet.zstd'))
        
        # Filter for just this target
        target_raw = raw_df.filter(pl.col('Target') == target_name)
        
        if len(target_raw) == 0:
            return {"data": []}
        
        # Apply additional filtering
        if filter_type == "platform" and filter_value != "all":
            target_raw = target_raw.filter(pl.col('platform') == filter_value)
        elif filter_type == "party" and filter_value != "all":
            target_raw = target_raw.filter(pl.col('Party') == filter_value)
        elif filter_type == "main_type" and filter_value != "all":
            target_raw = target_raw.filter(pl.col('MainType') == filter_value)
        
        print(f"Returning {len(target_raw)} raw data points for {target_name}")
        
        # Convert to list of dicts for JSON response
        return {"data": target_raw.to_dicts()}
    
    except Exception as e:
        print(f"Error loading raw data for {target_name}: {e}")
        return {"data": [], "error": str(e)}

@app.get("/filters")
async def get_filters(request: Request):
    """Get available filter options (requires authentication)"""
    # Authenticate the request
    await authenticate_request(request)
    return {
        "platforms": unique_platforms,
        "parties": unique_parties,
        "main_types": unique_main_types
    }

@app.get("/target/{target_name}/filters")
async def get_target_filters(
    request: Request,
    target_name: str
):
    """Get available filter options for a specific target (requires authentication)"""
    # Authenticate the request
    await authenticate_request(request)
    target_trends = all_trends_df.filter(pl.col('target') == target_name)
    
    available_platforms = target_trends.filter(
        pl.col('filter_type') == 'platform'
    )['filter_value'].unique().to_list()
    
    available_parties = target_trends.filter(
        pl.col('filter_type') == 'party'
    )['filter_value'].unique().to_list()
    
    available_main_types = target_trends.filter(
        pl.col('filter_type') == 'main_type'
    )['filter_value'].unique().to_list()
    
    return {
        "platforms": available_platforms,
        "parties": available_parties,
        "main_types": available_main_types
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)