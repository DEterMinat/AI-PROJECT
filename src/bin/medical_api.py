#!/usr/bin/env python3
"""
Medical API Console Script
Wrapper for running the FastAPI application with uvicorn
"""

import sys
import os
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

def main():
    """Main entry point for the medical-api command"""
    import uvicorn
    from src.api.integrated_medical_api import app

    # Default configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"

    print("🏥 Starting Medical AI API Server...")
    print(f"📍 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🔄 Reload: {reload}")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🩺 Health Check: http://localhost:8000/health")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)

    uvicorn.run(
        "src.api.integrated_medical_api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()