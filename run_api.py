#!/usr/bin/env python3
"""
Medical AI API Runner
=====================

Convenient script to run the FastAPI medical AI service.

Usage:
    python run_api.py [--host HOST] [--port PORT] [--reload]

Options:
    --host HOST     Host to bind to (default: 0.0.0.0)
    --port PORT     Port to bind to (default: 8000)
    --reload        Enable auto-reload for development
"""

import argparse
import uvicorn
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.api.integrated_medical_api import app


def main():
    parser = argparse.ArgumentParser(description="Run Medical AI API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    print("🏥 Starting Medical AI API...")
    print(f"📍 Host: {args.host}")
    print(f"🔌 Port: {args.port}")
    print(f"🔄 Reload: {args.reload}")
    print("=" * 50)

    uvicorn.run(
        "src.api.integrated_medical_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()