from src.cv import cv
from src.eval import evaluate_cv
from src.utils import HPARAMS
from utils import FCS_COL_NAMES, PEDIGREES, get_ms_files_df


path = '/mnt/data-ssd/cyto/fastcyto_dirs/extracted_stuff_big'
idxs = [i for i in range(4)]
probe = 'L'
pedigree_name = 'total'  # small total
folds = 2

hparams = HPARAMS
hparams.update({'iterations': 200})

for parent, kids in PEDIGREES[pedigree_name]:
    gate_panel = '_'.join(kids)
    print(f'CV {gate_panel}')

    df = get_ms_files_df(path, idxs, probe, gate_panel)

    vocab = kids if parent in ['NK', 'T'] and pedigree_name == 'total' else kids + ['rest']
    ds_kwargs = {'pre_gate': parent, 'vocab': vocab, 'fcs_col_names': FCS_COL_NAMES}
    result = cv(df, HPARAMS, folds=folds, ds_kwargs=ds_kwargs)

    result_dir = f'/home/lfisch/Projects/gatenet/data/results/ms/{probe}/{gate_panel}'
    evaluate_cv(result, result_dir)
