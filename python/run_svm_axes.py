"""Entry point: SVM separating axes in pre 5D s-dPCA space."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.dpca.svm_axes import main

if __name__ == "__main__":
    main()
