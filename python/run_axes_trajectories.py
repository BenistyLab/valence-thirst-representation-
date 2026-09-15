"""Entry point: pre-phase PC1–PC2 axes trajectories."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.dpca.axes_trajectories import main

if __name__ == "__main__":
    main()
