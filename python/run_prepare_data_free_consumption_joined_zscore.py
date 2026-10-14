import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib.prep.prepare_data_free_consumption_joined_zscore as prepare_data

# define cap here, usually 500 or 10,000,000
CAP = 10_000_000
prepare_data.DEFAULT_CAP = CAP

if __name__ == '__main__':
    prepare_data.main()
