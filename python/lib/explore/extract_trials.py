import numpy as np
import pandas as pd

# add SLM
NUMBER_TRIAL_TYPES = {
    3: 'water',
    4: 'nacl',
    7: 'water-free',
    8: 'water(s)',
    18: 'airpuff',
    26: 'pre-cue-water-SLMhigh(pav)'
}

def create_trial_boundaries(path):
    with np.load(path) as data:
        trial_types = data['trial_type']
        trial_lengths = np.array(data['segment_lengths'])
        trial_start_idx = list(np.cumsum(trial_lengths))
        trial_start_idx_copy = trial_start_idx.copy()

        trial_start_idx = [0] + trial_start_idx
        trial_start_idx.remove(trial_start_idx[-1])

        # trial_end_idx = [x - 1 for x in trial_start_idx_copy]
        trial_end_idx = trial_start_idx_copy

    start_end = []
    for a, b in zip(trial_start_idx, trial_end_idx):
        start_end.append([a, b])

    trial_boundaries = {'water': [], 'nacl': [], 'airpuff': [], 'water-free': [], 'water(s)': [], 'pre-cue-water-SLMhigh(pav)': []}
    for a, b in zip(list(trial_types), start_end):
        trial_boundaries[NUMBER_TRIAL_TYPES[a]].append(b)

    return trial_boundaries

def create_trial_labelled_df(path):
    """
    Receives path of npz file from prepare_data, and adds multiindex that labels the frames so you can see the frame,
    trial_type, and trial.
    For example:
        - call an airpuff trial like this: df['airpuff']
        - call second airpuff trial like this: df['airpuff',2]
    return:
    df
    trial_boundaries - a dict that maps the trial boundaries of each trial type
    """
    trial_boundaries = create_trial_boundaries(path)

    with np.load(path) as data:
        raw_data = data['concat']
    df = pd.DataFrame(raw_data)
    df.index.name = "Neuron"

    trial_type_labels = []
    trial_number_labels = []

    for frame in df.columns:
        trial_type = None
        trial_number = None

        for type_, ranges in trial_boundaries.items():
            for num, (start, end) in enumerate(ranges, start=1):
                if start <= frame <= end:
                    trial_type = type_
                    trial_number = num
                    break
            if trial_type is not None:
                break

        trial_type_labels.append(trial_type)
        trial_number_labels.append(trial_number)

    df.columns = pd.MultiIndex.from_arrays(
        [trial_type_labels, trial_number_labels, df.columns],
        names=['trial_type', 'trial', 'frame']
    )

    return trial_boundaries, df