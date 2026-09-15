import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.dpca.condition_distances import run_all_condition_distances
from lib.dpca.cross_run_decode import main as run_cross_decode
from lib.dpca.rank_diagnostics import main as run_rank_diagnostics
from lib.dpca.thirst_valence_axes import main as run_thirst_valence_axes

if __name__ == '__main__':
    run_all_condition_distances()
    run_thirst_valence_axes()
    run_cross_decode()
    run_rank_diagnostics()
