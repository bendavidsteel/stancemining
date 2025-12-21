from typing import List, Optional, Dict, Any
import datetime
import glob
import os
import logging
import random
import re
import secrets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('stancemining')

import llama_cpp
import numpy as np
import polars as pl
from pydantic import BaseModel
import pynndescent
import requests
import sklearn.metrics.pairwise
import sklearn.decomposition
from tqdm import tqdm
from pacmap import PaCMAP

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
    mean_stance: Optional[float] = None
    stance_std: Optional[float] = None
    stance_abs: Optional[float] = None
    n_favor: Optional[int] = None
    n_against: Optional[int] = None
    n_neutral: Optional[int] = None

class UmapResponse(BaseModel):
    data: List[UmapPoint]

class PCAStreamplotPoint(BaseModel):
    x: float
    y: float
    createtime: str
    filter_value: str
    platform: str
    year: int
    flow_u: Optional[float] = None
    flow_v: Optional[float] = None

class PCAStreamplotResponse(BaseModel):
    data: List[PCAStreamplotPoint]
    pca_info: Dict[str, Any]

# Global variables to store loaded data
target_df = None
all_trends_df = None
target_embeddings_df = None
embedding_model = None
nn_index = None

def load_env_vars():
    global STANCE_AUTH_URL_PATH, DATA_DIR_PATH
    # Load environment variables
    STANCE_AUTH_URL_PATH = os.getenv("AUTH_URL_PATH")
    DATA_DIR_PATH = os.getenv('DATA_DIR_PATH')
    if '~' in DATA_DIR_PATH:
        DATA_DIR_PATH = os.path.expanduser(DATA_DIR_PATH)

# --- Authentication Helper Functions ---
def get_token(username: str, password: str):
    """Get authentication token from the external API"""
    try:
        res = requests.get(
            STANCE_AUTH_URL_PATH, 
            params={"username": username, "password": password}, 
            verify=True
        )
        
        if res.status_code != 200:
            return None
            
        token = res.json().get("access_token")
        return token
    except Exception as e:
        logger.error(f"Error getting token: {e}")
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
        logger.error(f"Error verifying token: {e}")
        return False
    

def compute_target_embeddings(target_count_df: pl.DataFrame):
    """Compute embeddings for targets efficiently"""
    
    embeddings_path = os.path.join(DATA_DIR_PATH, 'cache', 'target_embeddings.parquet.zstd')
    
    # Initialize embeddings dataframe
    embeddings_df = None
    
    # Load cached embeddings if they exist
    if os.path.exists(embeddings_path):
        logger.info(f"Loading cached embeddings from {embeddings_path}")
        embeddings_df = pl.read_parquet(embeddings_path)
        logger.info(f"Loaded {len(embeddings_df)} cached embeddings")
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
        logger.info(f"Computing embeddings for {len(missing_targets)} new targets...")
        result = embedding_model.create_embedding(ProgressList(missing_targets))
        new_embeddings = np.stack([np.array(o['embedding']) for o in result['data']])
        
        # Create dataframe for new embeddings
        new_embeddings_df = pl.DataFrame({
            'Target': missing_targets,
            'Embedding': new_embeddings
        }, schema_overrides={'Embedding': pl.Array(pl.Float32, new_embeddings.shape[1])})
        
        # Combine with existing embeddings
        embeddings_df = pl.concat([embeddings_df, new_embeddings_df])
        
        # Save the complete embeddings
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
        embeddings_df.write_parquet(
            embeddings_path,
            compression="zstd"
        )
        logger.info(f"Updated target embeddings saved to {embeddings_path}")
    else:
        logger.info("All targets already have embeddings")

    # Join embeddings back to the target_count_df
    result_df = target_count_df.join(
        embeddings_df,
        on="Target",
        how="left"
    )
    
    return result_df


def compute_umap_embeddings(target_df: pl.DataFrame, target_embeddings_df: pl.DataFrame):
    """Compute UMAP embeddings for visualization with optimized memory usage"""

    umap_path = os.path.join(DATA_DIR_PATH, 'cache', 'umap_embeddings.parquet.zstd')
    # Check if UMAP embeddings already exist
    if os.path.exists(umap_path):
        logger.info(f"Loading cached UMAP embeddings from {umap_path}")
        umap_df = pl.read_parquet(umap_path)
        logger.info(f"Loaded {len(umap_df)} UMAP embeddings")
        return umap_df

    logger.info("Computing UMAP embeddings...")
    
    # Extract target names and get corresponding embeddings
    valid_embeddings_df = target_df.join(target_embeddings_df, on='Target')
    target_names = valid_embeddings_df['Target'].to_list()
    embeddings = valid_embeddings_df['Embedding'].to_numpy()
    
    # Set up UMAP with parameters suitable for visualization
    
    reducer = PaCMAP(
        n_components=2,
        random_state=42,
        verbose=True
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
    umap_df = umap_df.join(target_df.select(cols_to_join), on='Target', how='left')
    
    # Add stance statistics in a single join operation
    umap_df = umap_df.join(
        target_df,
        on='Target', 
        how='left'
    )

    

    # Save UMAP embeddings
    umap_df.write_parquet(umap_path, compression="zstd")
    logger.info(f"UMAP embeddings saved to {umap_path}")


async def authenticate_request(request: Request):
    """Helper function to authenticate a request with flexibility for development mode"""
    # Handle authentication (with development mode flexibility)
    try:
        # Check if we're in development mode with authentication skipping enabled
        skip_auth = os.getenv("REACT_APP_SKIP_AUTH") == "true" or not STANCE_AUTH_URL_PATH
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
        logger.error(f"Authentication error: {e}")
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

# --- Authentication Routes ---
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login endpoint that forwards authentication to the external API"""
    if STANCE_AUTH_URL_PATH is not None:
        token = get_token(form_data.username, form_data.password)
        
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
    else:
        token = secrets.token_urlsafe()
    
    return {"access_token": token, "token_type": "bearer"}

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    # In a real implementation, you might want to make an API call to get user details
    return User(
        username=current_user.username,
        full_name="Authenticated User",
    )

class ProgressList(list):
    """A list that shows progress bars when iterating."""
    
    def __init__(self, iterable):
        super().__init__(iterable)
    
    def __iter__(self):
        """Override default iteration to show progress."""
        # Use super().__iter__() to get the actual list iterator
        total = len(self)
        i = 0
        next_log_per = 0.1
        for item in super().__iter__():
            yield item
            i += 1
            if i / total >= next_log_per:
                logger.info(f"Progress: {i / total * 100:.1f}%")
                next_log_per += 0.1

@app.on_event("startup")
async def startup_event():
    """Load all precomputed data on startup"""
    global target_df, all_trends_df, target_embeddings_df, filter_types, embedding_model
    
    load_env_vars()

    logger.info("Loading precomputed data...")
    
    try:
        # Initialize embedding model
        logger.info("Initialized semantic search model")
        # try:
        embedding_model = llama_cpp.Llama.from_pretrained(
            repo_id="ChristianAzinn/gist-small-embedding-v0-gguf",
            filename="gist-small-embedding-v0.Q8_0.gguf",
            embedding=True
        )

        # Load filtered targets list
        stance_dir_path = os.path.join(DATA_DIR_PATH, 'doc_stance')
        if not os.path.exists(stance_dir_path):
            raise FileNotFoundError(f"Stance data directory must be accessible at {stance_dir_path}, only found files {os.listdir(DATA_DIR_PATH)} at {DATA_DIR_PATH}")
        stance_file_paths = glob.glob(os.path.join(stance_dir_path, '*.parquet.zstd'))
        if len(stance_file_paths) == 0:
            raise FileNotFoundError(f"No stance files found in {stance_dir_path}, some must be present.")
        try:
            stance_df = pl.read_parquet(stance_file_paths, columns=['Targets', 'Stances'])
        except Exception as ex:
            stance_df = pl.concat([pl.read_parquet(fp, columns=['Targets', 'Stances']) for fp in stance_file_paths], how='diagonal_relaxed')
        stance_df = stance_df.with_columns([pl.col('Targets').cast(pl.List(pl.String)), pl.col('Stances').cast(pl.List(pl.Int64))])
        if len(stance_df) == 0:
            raise ValueError("No stance data found in the provided directory")
        # Compute target counts
        target_df = stance_df.explode(['Targets', 'Stances'])\
            .drop_nulls('Targets')\
            .rename({'Targets': 'Target', 'Stances': 'Stance'})\
            .group_by('Target')\
            .agg([
                pl.len().alias('count'), 
                pl.col('Stance').mean().alias('mean_stance'),
                pl.col('Stance').std().alias('stance_std'),
                pl.col('Stance').abs().mean().alias('stance_abs'),
                pl.when(pl.col('Stance') == 1).then(1).otherwise(0).sum().alias('n_favor'),
                pl.when(pl.col('Stance') == -1).then(1).otherwise(0).sum().alias('n_against'),
                pl.when(pl.col('Stance') == 0).then(1).otherwise(0).sum().alias('n_neutral')
            ])
            
        min_count = 20
        target_df = target_df.filter(pl.col('count') >= min_count).sort('count', descending=True)

        # TODO filter or remove this line
        logger.info(f"Loaded {len(target_df)} valid targets with ≥{min_count} data points")

        # Load trends data
        target_trends_dir_path = os.path.join(DATA_DIR_PATH, 'target_trends')
        if not os.path.exists(target_trends_dir_path):
            raise FileNotFoundError(f"Target trends directory must be accessible at {target_trends_dir_path}")
        trend_file_paths = glob.glob(os.path.join(target_trends_dir_path, '*.parquet.zstd'))
        if len(trend_file_paths) == 0:
            raise FileNotFoundError(f"No trend files found in {target_trends_dir_path}, some must be present for the app to work")
        file_dfs = []
        for trend_file_path in tqdm(trend_file_paths, desc='Loading trend files'):
            try:
                file_df = pl.read_parquet(trend_file_path)
                if 'trend_mean' not in file_df.columns:
                    # dataframe is interpolator output, skip
                    continue
                file_dfs.append(file_df)
            except Exception as e:
                logger.error(f"Error loading trend file {trend_file_path}: {e}")
        all_trends_df = pl.concat(file_dfs)

        # only keep targets we have trends for
        target_df = target_df.join(all_trends_df.select('target').unique('target'), left_on='Target', right_on='target', how='right', maintain_order='left').rename({'target': 'Target'})

        logger.info(f"Loaded unified trends dataframe with {len(all_trends_df)} rows")

        # Note: Raw data will be loaded on-demand, not at startup
        
        # Load embeddings for semantic search
        target_embeddings_df = compute_target_embeddings(target_df)
        logger.info(f"Loaded embeddings for {len(target_embeddings_df)} targets")

        # Convert dictionary to a list of embeddings in the same order as all_targets
        embeddings_list = target_df.join(target_embeddings_df, on='Target')['Embedding'].to_numpy()
        
        try:
            # Create the NNDescent index
            global nn_index
            nn_index = pynndescent.NNDescent(embeddings_list)
            logger.info("Initialized PyNNDescent index for fast similarity search")
        except Exception as ex:
            logger.error(f"PyNNDescent failed with error: {ex}")

        unique_filter_types = all_trends_df['filter_type'].unique().to_list()
        logger.info(f"Loaded {len(unique_filter_types)} unique filter types: {unique_filter_types}")

        
        umap_df = compute_umap_embeddings(target_df, target_embeddings_df)
    
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        # Re-raise to prevent the app from starting with incomplete data
        raise e

@app.get("/umap", response_model=UmapResponse)
async def get_umap_data(request: Request):
    """Get UMAP visualization data (requires authentication)"""
    # Authenticate the request
    await authenticate_request(request)
    try:
        # Load UMAP embedding data
        umap_df = pl.read_parquet(os.path.join(DATA_DIR_PATH, 'cache', 'umap_embeddings.parquet.zstd'))
        
        if len(umap_df) > 10000:
            umap_df = umap_df.sort('count', descending=True).head(10000)

        # Convert to list of dicts for JSON response
        umap_data = umap_df.to_dicts()
        
        return {"data": umap_data}
    except Exception as e:
        logger.error(f"Error loading UMAP data: {e}")
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
        query_embedding = np.array(embedding_model.create_embedding([query])['data'][0]['embedding']).reshape(1, -1)
        
        if nn_index:
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
        else:
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

    filtered_trends = filtered_trends.drop_nans('trend_mean')
    
    if np.all(np.isnan(filtered_trends['trend_lower'].to_numpy())):
        filtered_trends = filtered_trends.with_columns(
            pl.col('trend_lower').fill_nan(pl.col('trend_mean')),
            pl.col('trend_upper').fill_nan(pl.col('trend_mean'))
        )

    # Convert to list of dicts for JSON response
    return {"data": filtered_trends.to_dicts()}

@app.get("/target/{target_name}/trends/batch")
async def get_target_trends(
    request: Request,
    target_name: str, 
    filter_type: str = Query(), 
    filter_values: str = Query()
):
    """Get trend data for a specific target with optional filtering (requires authentication)"""
    # Authenticate the request
    await authenticate_request(request)
    
    # Filter precomputed data for this target
    target_trends = all_trends_df.filter(pl.col('target') == target_name)
    
    if len(target_trends) == 0:
        return {"data": []}
    
    filter_values = filter_values.split(',')
    
    # Apply additional filtering
    filtered_trends = target_trends.filter((pl.col('filter_type') == filter_type) & (pl.col('filter_value').is_in(filter_values)))

    filtered_trends = filtered_trends.drop_nans()
    
    # Convert to list of dicts for JSON response
    return {"data": {g_vals[0]: df.to_dict(as_series=False) for (g_vals, df) in filtered_trends.partition_by('filter_value', as_dict=True).items()}}

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
        logger.info(f"Loading raw data for target: {target_name}")
        
        stance_dir_path = os.path.join(DATA_DIR_PATH, 'doc_stance')
        if not os.path.exists(stance_dir_path):
            raise FileNotFoundError(f"Stance data directory must be accessible at {stance_dir_path}, only found files {os.listdir(DATA_DIR_PATH)} at {DATA_DIR_PATH}")
        stance_file_paths = glob.glob(os.path.join(stance_dir_path, '*.parquet.zstd'))
        if len(stance_file_paths) == 0:
            raise FileNotFoundError(f"No stance files found in {stance_dir_path}, some must be present.")
        target_df = None
        for stance_file_path in stance_file_paths:
            try:
                file_stance_df = pl.read_parquet(stance_file_path)
                file_target_df = file_stance_df.explode(['Targets', 'Stances'])\
                    .drop_nulls('Targets')\
                    .filter(pl.col('Targets') == target_name)
                
                # Apply additional filtering
                if filter_value != "all":
                    file_target_df = file_target_df.filter(pl.col(filter_type) == filter_value)

                if len(file_target_df) == 0:
                    continue
                
                file_target_df = file_target_df.rename({'Targets': 'Target', 'Stances': 'Stance'})
                target_df = pl.concat([target_df, file_target_df], how='diagonal_relaxed') if target_df is not None else file_target_df
            except Exception as ex:
                logger.error(f"Error loading stance file {stance_file_path}: {ex}")

        if target_df is None:
            return {"data": []}

        logger.info(f"Returning {len(target_df)} raw data points for {target_name}")

        # Convert to list of dicts for JSON response
        return {"data": target_df.to_dicts()}
    
    except Exception as e:
        logger.error(f"Error loading raw data for {target_name}: {e}")
        return {"data": [], "error": str(e)}

@app.get("/filters")
async def get_filters(request: Request):
    """Get available filter options (requires authentication)"""
    # Authenticate the request
    await authenticate_request(request)
    return {filter_type: [] for filter_type in []}

@app.get("/target/{target_name}/filters")
async def get_target_filters(
    request: Request,
    target_name: str
):
    """Get available filter options for a specific target (requires authentication)"""
    # Authenticate the request
    await authenticate_request(request)
    target_trends = all_trends_df.filter(pl.col('target') == target_name)
    
    filters = target_trends.group_by('filter_type')\
        .agg(pl.col('filter_value'))\
        .with_columns(pl.col('filter_value').list.unique())\
        .to_dicts()

    return {f['filter_type']: f['filter_value'] for f in filters}


def compute_pca_streamplot_data(df, cols='both'):
    """Compute PCA analysis and flow field data for streamplot visualization"""
    try:
        # Pivot data
        logger.info("Pivoting data for PCA...")
        target_df = df.pivot(on='target', index=['createtime', 'filter_value'], values=['trend_mean', 'volume'])
        
        # Get feature columns
        feature_cols = [c for c in target_df.columns if c not in ['createtime', 'filter_value']]
        stance_cols = [c for c in feature_cols if 'trend_mean' in c]
        volume_cols = [c for c in feature_cols if 'volume' in c]
        
        # Handle missing values
        # sklearn simple imputer is very slow so use polars instead
        logger.info("Imputing missing values...")
        # replace completely missing values with mean
        target_df = target_df.with_columns(
            [pl.col(c).backward_fill().over('filter_value') for c in stance_cols]
        ).with_columns(
            [pl.col(c).forward_fill().over('filter_value') for c in stance_cols]
        ).with_columns(
            [pl.col(c).fill_null(strategy='mean') for c in stance_cols]
        ).with_columns(
            [pl.col(c).fill_null(0).rolling_mean(window_size=3).over('filter_value').fill_null(value=0) for c in volume_cols]
        )

        X = target_df.select(feature_cols).to_numpy()
        n_components = 2

        # Apply PCA
        logger.info("Applying PCA")
        try:
            import faiss

            pca = faiss.PCAMatrix(X.shape[1], n_components)
            pca.train(X)
            coords = pca.apply(X)

            explained_variance = None
            pca_components = components = faiss.vector_to_array(pca.A).reshape(n_components, X.shape[1])
        except ImportError:
            logger.warning("FAISS not available, using sklearn PCA instead")
            pca = sklearn.decomposition.PCA(n_components=n_components)
            coords = pca.fit_transform(X)

            explained_variance = pca.explained_variance_ratio_.tolist()
            pca_components = pca.components_

        # Add coordinates and extract metadata
        target_df = target_df.with_columns([
            pl.Series('x', coords[:, 0]),
            pl.Series('y', coords[:, 1]),
            pl.col('createtime').dt.year().alias('year'),
            pl.col('filter_value').str.split('-').list.get(1).alias('platform')
        ])

        logger.info("Calculating flow field...")
        # Calculate flow field (simplified version for web)
        flow_data = calculate_flow_field(target_df)
        
        # Get top contributing features
        pca_info = {
            'explained_variance_ratio': explained_variance,
            'feature_names': feature_cols,
            'components': pca_components
        }
        
        return target_df, pca_info, flow_data
        
    except Exception as e:
        logger.error(f"Error computing PCA data: {e}")
        return None, None, None

def calculate_flow_field(target_df: pl.DataFrame):
    """Calculate simplified flow field data for web visualization"""
    
    return target_df.select(['filter_value', 'x', 'y'])\
        .with_columns(pl.concat_arr(['x', 'y']).alias('coord'))\
        .with_columns((pl.col('coord') - pl.col('coord').shift(1).over('filter_value')).alias('flow'))\
        .select([
            pl.col('coord').arr.get(0).alias('x'),
            pl.col('coord').arr.get(1).alias('y'),
            pl.col('flow').arr.get(0).alias('u'),
            pl.col('flow').arr.get(1).alias('v'),
            pl.col('filter_value').alias('user_id')
        ]).to_dicts()
    # flow_data = []
    
    # # Group by user and calculate flow vectors
    # user_groups = target_df.group_by('filter_value')
    
    # for user_id, user_data in user_groups:
    #     if len(user_data) > 1:
    #         user_coords = user_data.select(['x', 'y']).to_numpy()
            
    #         for i in range(len(user_coords) - 1):
    #             start_pos = user_coords[i]
    #             end_pos = user_coords[i + 1]
    #             flow_vector = end_pos - start_pos
                
    #             # Skip very small movements
    #             if np.linalg.norm(flow_vector) > 0.01:
    #                 flow_data.append({
    #                     'x': float(start_pos[0]),
    #                     'y': float(start_pos[1]),
    #                     'u': float(flow_vector[0]),
    #                     'v': float(flow_vector[1]),
    #                     'user_id': user_id[0] if isinstance(user_id, tuple) else user_id
    #                 })
        

@app.get("/pca-streamplot", response_model=PCAStreamplotResponse)
async def get_pca_streamplot_data(request: Request):
    """Get PCA streamplot visualization data (requires authentication)"""
    await authenticate_request(request)
    
    try:
        # Load the stance target trends data
        df = all_trends_df.filter(pl.col('filter_type') == 'PlatformHandleID')
        if df is None:
            return {"data": [], "pca_info": {}}
        
        # Compute PCA and flow field data
        result_df, pca_info, flow_data = compute_pca_streamplot_data(df)
        if result_df is None:
            return {"data": [], "pca_info": {}}
        
        # Convert to response format
        data_points = result_df.select([
            pl.col('x'),
            pl.col('y'),
            pl.col('createtime').dt.to_string('iso'),
            pl.col('filter_value'),
            pl.col('platform').fill_null('unknown'),
            pl.col('year')
        ]).to_dicts()
        
        # Add flow data to pca_info
        pca_info['flow_data'] = flow_data
        
        return PCAStreamplotResponse(data=data_points, pca_info=pca_info)
        
    except Exception as e:
        logger.error(f"Error generating PCA streamplot data: {e}")
        return {"data": [], "pca_info": {"error": str(e)}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)