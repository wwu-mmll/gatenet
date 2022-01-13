from src.data.files import get_files_df
from src.data.utils import unique_labels
from src.cv import cv
from src.eval import evaluate_cv
from src.utils import HPARAMS, set_seed
from utils import LABEL_COL, flowcap_paths


set_seed()

flowcap_ds = 'GvHD'  # NDD Lymph GvHD StemCell CFSE
fcs_path, labels_path = flowcap_paths(flowcap_ds, '/mnt/data-ssd/cyto/flowcap1')
label_col = LABEL_COL

df = get_files_df(fcs_path, labels_path)  # , file_idxs=[i for i in range(4)])
vocab = unique_labels(labels_path, label_col)
ds_kwargs = {'vocab': vocab, 'multiclass_col': label_col}

hparams = HPARAMS
hparams.update({'iterations': 10000})

result = cv(df, hparams, folds=3, ds_kwargs=ds_kwargs)

result_dir = f'/home/lfisch/Projects/gatenet/data/results/flowcap/{flowcap_ds}'  # f'data/results/flowcap/{flowcap_ds}'
evaluate_cv(result, result_dir)
