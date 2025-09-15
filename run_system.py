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
    
    # Simplified check using the requirements.txt for consistency
    try:
        import pkg_resources
        with open('requirements.txt', 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        pkg_resources.require(requirements)
        logger.info("All dependencies are installed.")
        return True
    except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict) as e:
        logger.error(f"Dependency issue: {e}")
        logger.info("Please install/update packages using: pip install -r requirements.txt")
        return False
    except FileNotFoundError:
        logger.error("requirements.txt not found. Cannot check dependencies.")
        return False


def setup_directories():
    """Setup required directories"""
    logger.info("Setting up directories...")
    
    directories = [
        settings.DATA_DIR,
        settings.MODELS_DIR,
        settings.CONFIG_DIR,
        settings.LOGS_DIR,
        settings.DATA_DIR / "real_time_cache"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")
    
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
        # Use Popen to run in the background without blocking
        process = subprocess.Popen(cmd)
        return process
        
    except Exception as e:
        logger.error(f"Backend failed to start: {e}")
        return None

def run_frontend():
    """Run Streamlit frontend"""
    logger.info("Starting Streamlit frontend...")
    
    # Add a helpful, clear message for the user
    logger.info("="*60)
    logger.info("The system is starting the user interface.")
    logger.info("Please open your web browser and go to the following URL:")
    logger.info(f"    http://localhost:8501")
    logger.info("="*60)
    
    try:
        # Start frontend server
        cmd = [
            sys.executable, "-m", "streamlit",
            "run", "src/frontend/app.py",
            "--server.port", "8501",
            "--server.address", "127.0.0.1" # Listen on all interfaces
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
        # Assuming train_model.py is at the root
        cmd = [sys.executable, "train_model.py", "--model", "all"]
        
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
    
    if not check_dependencies(): return
    setup_directories()
    
    # Check if models exist, if not, train them
    # A more robust check might be for a training report or a specific model file
    if not (settings.MODELS_DIR / "xgboost_model.pkl").exists():
        logger.warning("Models not found. Starting initial training...")
        run_training()
    
    logger.info("System ready. Starting background and foreground services...")
    
    backend_process = None
    try:
        # Start backend in the background
        backend_process = run_backend()
        
        # Wait for backend to initialize
        logger.info("Waiting for backend to initialize (5s)...")
        time.sleep(5)
        
        # Start frontend in the foreground (this will block)
        run_frontend()
        
    finally:
        if backend_process:
            logger.info("Shutting down backend service...")
            backend_process.terminate()
            backend_process.wait()


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
            if check_dependencies():
                setup_directories()
                logger.info("System setup completed")
        
        elif args.command == 'backend':
            if check_dependencies():
                setup_directories()
                process = run_backend()
                if process:
                    process.wait() # Wait for Ctrl+C
        
        elif args.command == 'frontend':
            if check_dependencies():
                setup_directories()
                run_frontend()
        
        elif args.command == 'train':
            if check_dependencies():
                setup_directories()
                run_training()
        
        elif args.command == 'test':
            if check_dependencies():
                run_tests()
        
        elif args.command == 'full':
            run_full_system()
            
    except KeyboardInterrupt:
        logger.info("System stopped by user.")
    except Exception as e:
        logger.error(f"A system error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
