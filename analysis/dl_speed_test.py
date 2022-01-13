from tqdm import tqdm

from src.data import files, set, loader, utils
from utils import LABEL_COL, flowcap_paths


flowcap_ds = 'GvHD'
fcs_path, labels_path = flowcap_paths(flowcap_ds, '/mnt/data-ssd/cyto/flowcap1')
label_col = LABEL_COL

df = files.get_files_df(fcs_path, labels_path)
vocab = utils.unique_labels(labels_path, label_col)
ds_kwargs = {'vocab': vocab, 'multiclass_col': label_col}
ds = set.GatingDataset(df, multiclass_col=label_col, vocab=vocab)
dl = loader.GateNetLoader(ds, drop_last=False)

for b in tqdm(dl):
    print('', end='', flush=True)
