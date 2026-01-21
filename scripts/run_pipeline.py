"""
Pipeline Runner Script

Executes the full Fall Detection pipeline:
1. Inference (Batch processing of dataset)
2. Evaluation (Metric calculation and plotting)

Usage:
    python scripts/run_pipeline.py [--limit 10] [--no-video]
"""

import argparse
import subprocess
import sys
from pathlib import Path
from loguru import logger

def run_command(command, description):
    """Run a shell command with logging."""
    logger.info(f"STARTING: {description}")
    logger.info(f"Command: {' '.join(command)}")
    
    try:
        # Use shell=True for Windows compatibility if needed, but list args are safer
        # On Windows, python is usually needed explicitly
        result = subprocess.run(command, check=True, text=True)
        logger.success(f"COMPLETED: {description}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FAILED: {description}")
        logger.error(f"Exit Code: {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run Full Fall Detection Pipeline")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of sequences for inference")
    parser.add_argument("--no-video", action="store_true", help="Skip video generation (faster)")
    parser.add_argument("--skip-infer", action="store_true", help="Skip inference, run eval only")
    
    args = parser.parse_args()
    
    # Paths
    base_dir = Path(__file__).parent.parent
    script_infer = base_dir / "scripts" / "run_infer.py"
    script_eval = base_dir / "scripts" / "run_eval.py"
    
    python_exe = sys.executable
    
    # 1. Run Inference
    if not args.skip_infer:
        cmd_infer = [python_exe, str(script_infer)]
        if args.limit > 0:
            cmd_infer.extend(["--limit", str(args.limit)])
        
        # run_infer.py defaults to generating video. 
        # Pass --no-video only if user requested it.
        if args.no_video:
            cmd_infer.append("--no-video")

        if not run_command(cmd_infer, "Inference Step"):
            sys.exit(1)
            
    # 2. Run Evaluation
    cmd_eval = [python_exe, str(script_eval)]
    if not run_command(cmd_eval, "Evaluation Step"):
        sys.exit(1)
        
    logger.success("PIPELINE COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    main()
