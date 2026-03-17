# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StanceMining is a library for stance detection and target extraction from text corpora. It provides an end-to-end pipeline for:
- Extracting stance targets (noun-phrases or claims) from documents
- Detecting stance (favor/against/neutral) towards targets
- Clustering targets into higher-level topics
- Inferring stance trends over time via Gaussian processes or kernel regression
- Visualizing results through a web dashboard

## Common Commands

### Running Tests
```bash
pytest                           # Run all tests
pytest tests/test_main.py        # Run specific test file
pytest tests/test_main.py::test_filter_targets -v  # Run single test
```

### Linting
```bash
ruff check .                     # Run linter (uses F, D rules with Google docstring convention)
```

### Building Documentation
```bash
cd docs && sphinx-build -b html . _build/html
```

### Training Models
```bash
# Copy and configure config file first
cp ./config/config_default.yaml ./config/config.yaml
python ./experiments/scripts/train_model.py
```

### Running the Web App
```bash
export STANCE_DATA_PATH=<your-data-path>
docker compose -f ./app/compose.yaml up
```

### Installing the Package
```bash
pip install stancemining
# Or with extras: pip install stancemining[gp,train,plot]
```

## Architecture

### Core Library (`stancemining/`)
- `main.py`: `StanceMining` class - main entry point with `fit_transform()` for the full pipeline
- `llms.py`: LLM wrappers (`Transformers`, `VLLM`, `Anthropic`) for inference
- `finetune.py`: Model fine-tuning with LoRA, supports classification and generation tasks
- `estimate.py`: Time series trend inference using Gaussian processes (gpytorch/pyro) or kernel regression
- `prompting.py`: Zero-shot prompting utilities
- `utils.py`: Target filtering, embedding utilities, similarity functions

### Pipeline Flow
1. **Target Extraction**: Extract noun-phrases or claims from documents via fine-tuned models or prompting
2. **Target Deduplication**: Filter similar targets using embedding similarity (default threshold 0.8)
3. **Higher-level Targets**: Cluster targets using BERTopic or Toponymy topic models
4. **Stance Detection**: Classify stance per document-target pair
5. **Trend Inference**: Fit GP or kernel regression models for stance time series

### Task Types
- Classification tasks: `stance-classification`, `claim-entailment-{2,3,4,5,7}way`
- Generation tasks: `topic-extraction`, `claim-extraction`

### Web App (`app/`)
- `backend/`: FastAPI server serving trends, UMAP visualizations, semantic search
- `frontend/`: React dashboard for exploring stance trends
- Data expected in `STANCE_DATA_PATH` with `doc_stance/` and `target_trends/` directories containing parquet files

## Key Patterns

### LLM Inference
Default uses vLLM with fine-tuned SmolLM2-360M models hosted on HuggingFace (`bendavidsteel/SmolLM2-360M-Instruct-*`). Falls back to transformers if vLLM unavailable.

### Embeddings
Uses `intfloat/multilingual-e5-small` by default. Embeddings are cached in a polars DataFrame with `text` and `embedding` columns.

### GPU Acceleration
Uses cuml for UMAP/HDBSCAN clustering, gpytorch for Gaussian processes. Falls back to CPU implementations when CUDA unavailable.

## Python Code Style
- Prefer polars over pandas for dataframes
- Prefer backslash over parentheses for multi-line calls
- Prefer vLLM over HF transformers for inference
- Prefer GPU implementations (cuml, cugraph) when available
- Import blocks: stdlib, then third-party (general imports), then library-specific imports
- Avoid for loops with numpy/jax - use vectorized operations
- Avoid global variables

## Performance Expectations

### Trend Inference (`estimate.py`)
For 1000 training samples, 100 test points, 100 bootstrap samples:
- Original kernel regression (CPU): ~1.2s
- Bayesian KRR (GPU batched): ~1.2s
- Exact GP (GPU, 50 iter): ~1.6s
- Bayesian KRR (numba CPU): ~2.1s
- Bayesian KRR (CPU batched numpy): ~3.3s

Ordinal GP with SVI takes minutes - avoid for large-scale inference.

## Gotchas

### Lengthscale Parameterization
The GP functions use log-normal priors for lengthscale. `lengthscale_loc` is passed to `torch.log()` internally:
- `lengthscale_loc=2.0, lengthscale_scale=0.1` gives mode ≈ 7.3 months
- Mode formula: `exp(log(loc) - scale²)` = `exp(ln(2.0) - 0.01)` ≈ 7.3
- This is designed for monthly time scales (`time_scale='1mo'`)

### GPU Batched Broadcasting
When implementing batched GPU operations, be careful with tensor shapes:
- Training kernel: `(n_bootstrap, n_samples, n_samples)`
- Test kernel: `(n_bootstrap, n_test, n_samples)`
- Use `timestamps.unsqueeze(1)` not `timestamps.unsqueeze(-1).unsqueeze(1)` for test kernel broadcasting

### Bayesian KRR Alpha Parameter
The `alpha` parameter in Bayesian KRR controls prior strength toward 0:
- `alpha=0.01`: Weak prior, predictions close to data (can overfit)
- `alpha=1.0`: Moderate prior, good default
- `alpha=5.0`: Strong prior, heavy shrinkage toward 0
