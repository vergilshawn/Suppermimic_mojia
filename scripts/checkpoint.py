"""Checkpoint command entry point."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_control.commands.checkpoint import main

if __name__ == "__main__":
    main()
