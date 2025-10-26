#!/usr/bin/env python3
"""
Medical AI ML Pipeline Runner
============================

Convenient script to run machine learning pipeline components.

Usage:
    python run_ml_pipeline.py <pipeline_step> [options]

Pipeline Steps:
    data_collection     - Collect and prepare medical data
    data_cleaning       - Clean and preprocess data
    eda                 - Exploratory data analysis
    feature_engineering - Feature engineering and selection
    data_splitting      - Split data for training/validation/testing
    model_selection     - Compare different models
    model_training      - Train the selected model
    model_evaluation    - Evaluate model performance

Options:
    --config FILE       Configuration file (default: config/config.json)
    --output DIR        Output directory (default: data/)
    --verbose           Enable verbose output
"""

import argparse
import sys
import importlib
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Pipeline step mapping
PIPELINE_STEPS = {
    "data_collection": "src.ml_pipeline.data_collection",
    "data_cleaning": "src.ml_pipeline.data_cleaning",
    "eda": "src.ml_pipeline.eda",
    "feature_engineering": "src.ml_pipeline.feature_engineering",
    "data_splitting": "src.ml_pipeline.data_splitting",
    "model_selection": "src.ml_pipeline.model_selection",
    "model_training": "src.ml_pipeline.model_training",
    "model_evaluation": "src.ml_pipeline.model_evaluation",
}


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    import json
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  Config file not found: {config_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in config file: {e}")
        return {}


def run_pipeline_step(step_name: str, config: Dict[str, Any], verbose: bool = False):
    """Run a specific pipeline step."""
    if step_name not in PIPELINE_STEPS:
        print(f"❌ Unknown pipeline step: {step_name}")
        print("Available steps:", ", ".join(PIPELINE_STEPS.keys()))
        return False

    module_path = PIPELINE_STEPS[step_name]

    try:
        if verbose:
            print(f"🔄 Importing {module_path}...")

        module = importlib.import_module(module_path)

        # Try to find main function
        if hasattr(module, 'main'):
            if verbose:
                print(f"🚀 Running {step_name}...")
            result = module.main(config)
            if verbose:
                print(f"✅ {step_name} completed successfully")
            return result
        else:
            print(f"❌ No main function found in {module_path}")
            return False

    except ImportError as e:
        print(f"❌ Failed to import {module_path}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error running {step_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Medical AI ML Pipeline")
    parser.add_argument("step", choices=PIPELINE_STEPS.keys(),
                       help="Pipeline step to run")
    parser.add_argument("--config", default="config/config.json",
                       help="Configuration file path")
    parser.add_argument("--output", default="data/",
                       help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")

    args = parser.parse_args()

    print("🔬 Medical AI ML Pipeline")
    print(f"📋 Step: {args.step}")
    print(f"⚙️  Config: {args.config}")
    print(f"📁 Output: {args.output}")
    print("=" * 50)

    # Load configuration
    config = load_config(args.config)
    if not config and args.step != "data_collection":
        print("⚠️  Using default configuration")

    # Add output directory to config
    config['output_dir'] = args.output

    # Run the pipeline step
    success = run_pipeline_step(args.step, config, args.verbose)

    if success:
        print("✅ Pipeline step completed successfully!")
    else:
        print("❌ Pipeline step failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()