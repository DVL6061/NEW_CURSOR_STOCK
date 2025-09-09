"""
Main system runner for the Enterprise Stock Forecasting System
"""

import argparse
import asyncio
import logging
import subprocess
import sys
from pathlib import Path
import time

from config.settings import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed"""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'yfinance', 'pandas', 'numpy', 'sklearn', 'xgboost',
        'torch', 'transformers', 'streamlit', 'fastapi', 'uvicorn',
        'plotly', 'shap', 'requests', 'bs4'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Please install missing packages using: pip install -r requirements.txt")
        return False
    
    logger.info("All dependencies are installed")
    return True

def setup_directories():
    """Setup required directories"""
    logger.info("Setting up directories...")
    
    directories = [
        settings.DATA_DIR,
        settings.MODELS_DIR,
        settings.CONFIG_DIR,
        settings.DATA_DIR / "real_time_cache"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    logger.info("Directory setup completed")

def run_backend():
    """Run FastAPI backend"""
    logger.info("Starting FastAPI backend...")
    
    try:
        # Start backend server
        cmd = [
            sys.executable, "-m", "uvicorn",
            "src.backend.main:app",
            "--host", settings.API_HOST,
            "--port", str(settings.API_PORT),
            "--reload"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Backend failed to start: {e}")
        raise
    except KeyboardInterrupt:
        logger.info("Backend stopped by user")

def run_frontend():
    """Run Streamlit frontend"""
    logger.info("Starting Streamlit frontend...")
    
    try:
        # Start frontend server
        cmd = [
            sys.executable, "-m", "streamlit",
            "run", "src/frontend/app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Frontend failed to start: {e}")
        raise
    except KeyboardInterrupt:
        logger.info("Frontend stopped by user")

def run_training():
    """Run model training"""
    logger.info("Starting model training...")
    
    try:
        # Run training script
        cmd = [sys.executable, "train_models.py", "--model", "all"]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed: {e}")
        raise

def run_tests():
    """Run test suite"""
    logger.info("Running test suite...")
    
    try:
        # Run pytest
        cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Tests failed: {e}")
        raise

def run_full_system():
    """Run the complete system"""
    logger.info("Starting Enterprise Stock Forecasting System...")
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Setup directories
    setup_directories()
    
    # Check if models exist, if not, train them
    if not (settings.MODELS_DIR / "xgboost_model.pkl").exists():
        logger.info("Models not found. Starting training...")
        run_training()
    
    logger.info("System ready. Starting services...")
    
    # Start backend in background
    import threading
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    
    # Wait for backend to start
    time.sleep(5)
    
    # Start frontend
    run_frontend()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Enterprise Stock Forecasting System")
    parser.add_argument("command", choices=[
        'backend', 'frontend', 'train', 'test', 'full', 'setup'
    ], help="Command to run")
    parser.add_argument("--verbose", action='store_true', help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.command == 'setup':
            check_dependencies()
            setup_directories()
            logger.info("System setup completed")
        
        elif args.command == 'backend':
            check_dependencies()
            setup_directories()
            run_backend()
        
        elif args.command == 'frontend':
            check_dependencies()
            setup_directories()
            run_frontend()
        
        elif args.command == 'train':
            check_dependencies()
            setup_directories()
            run_training()
        
        elif args.command == 'test':
            check_dependencies()
            run_tests()
        
        elif args.command == 'full':
            run_full_system()
        
    except KeyboardInterrupt:
        logger.info("System stopped by user")
    except Exception as e:
        logger.error(f"System error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
