"""Entry point: fixed thirst / valence axes from pre s-space."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.dpca.thirst_valence_axes import main

if __name__ == "__main__":
    main()
