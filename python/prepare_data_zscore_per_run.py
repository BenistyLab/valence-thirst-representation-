import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.prep.prepare_data_score_per_run import main

if __name__ == '__main__':
    main()
