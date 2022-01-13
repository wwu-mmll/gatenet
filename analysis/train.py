from fastai.data.all import torch

from src.data.files import get_files_df
from src.data.utils import unique_labels
from src.train import train
from src.utils import HPARAMS, set_seed
from utils import LABEL_COL, flowcap_paths


set_seed()

flowcap_ds = 'NDD'  # NDD Lymph GvHD StemCell CFSE
fcs_path, labels_path = flowcap_paths(flowcap_ds, '/mnt/data-ssd/cyto/flowcap1')
label_col = LABEL_COL

df = get_files_df(fcs_path, labels_path)
vocab = unique_labels(labels_path, label_col)
ds_kwargs = {'vocab': vocab, 'multiclass_col': label_col}

hparams = HPARAMS
hparams.update({'epochs': 5})
learner = train(df.iloc[:-3], df.iloc[-3:], hparams, ds_kwargs=ds_kwargs)

model_path = f'/home/lfisch/Projects/gatenet/data/models/{flowcap_ds}.pth'
torch.save(learner.model.state_dict(), model_path)
