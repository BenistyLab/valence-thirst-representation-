import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.lda import decode, plots

if __name__ == '__main__':
    decode.main()
    plots.main()
