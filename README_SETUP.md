# Medical AI System Setup Guide

## Prerequisites

1.  **Conda**: Install [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2.  **Docker** (Optional): Required for N8N workflow.

## Quick Setup (Windows)

1.  Run `SETUP_CONDA.bat`.
    *   This will create the `medical-ai` environment.
    *   It will install all dependencies.
    *   It will install the project in editable mode.

## Running the System

1.  Run `START-ALL.bat`.
    *   This will check for the Conda environment.
    *   It will start the API and (optionally) N8N.

## Manual Setup

If you prefer to set up manually:

```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate medical-ai

# Install project
pip install -e .
```

## Project Structure

*   `src/`: Source code
    *   `api/`: FastAPI application
    *   `ml_pipeline/`: Machine learning scripts
    *   `models/`: Model definitions
*   `scripts/`: Utility scripts
*   `config/`: Configuration files
*   `data/`: Data storage
