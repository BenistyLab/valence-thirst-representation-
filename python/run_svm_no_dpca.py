"""Entry point: SVM separating axes on normalized trials (no dPCA)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.dpca.svm_no_dpca import main

if __name__ == "__main__":
    main()
