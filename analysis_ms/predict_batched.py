import sys, os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from tqdm import tqdm

from src.utils import set_seed
from analysis_ms.predict import predict_ms
from analysis_ms.utils import PEDIGREES, get_batch_idxs


def predict_ms_batched(batch_idxs, dirs: dict, probe: str, pedigree: tuple):
    for b_idxs in tqdm(batch_idxs):
        sys.stdout = open(os.devnull, 'w')
        predict_ms(dirs, b_idxs, probe, pedigree)
        sys.stdout = sys.__stdout__


if __name__ == '__main__':
    set_seed()

    probe = 'L'
    skip_idxs = {'L': [8267], 'B': [9380, 9419]}[probe]
    event_count_dir = '/home/lfisch/Projects/gatenet/analysis_ms/archive'
    event_counts = pd.read_csv(f'{event_count_dir}/event_count_{probe}.csv', index_col=[0]).Event_Count.tolist()
    batch_idxs = get_batch_idxs(event_counts, 3*10**6, skip_idxs)
    # batch_idxs[0] = [i for i in range(10)]
    pedi_name = 'total'  # small total
    pedi = PEDIGREES[pedi_name]

    lmd_11k_dir = '/mnt/data-ssd/cyto/lmd_11k'
    pred_11k_dir = '/mnt/data-ssd/cyto/predicted_labels_11k'

    pred_dir = pred_11k_dir  # pred_11k_dir '/home/lfisch/Projects/gatenet/data/predictions/ms'
    dirs = {'fcs': lmd_11k_dir,  # '/mnt/data-ssd/cyto/fastcyto_dirs/extracted_stuff_big/lmds',
            'labels': None,
            'model': f'/home/lfisch/Projects/gatenet/data/models/ms/pedigree_{pedi_name}/{probe}',
            'gates': f'{pred_dir}/pedigree_{pedi_name}/{probe}/argmax_preds',
            'preds': f'{pred_dir}/pedigree_{pedi_name}/{probe}/preds'}

    predict_ms_batched(batch_idxs, dirs, probe, pedi)
