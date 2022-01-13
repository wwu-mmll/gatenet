from src.data.files import get_files_df
from src.data.utils import unique_labels
from src.predict import predict
from src.utils import HPARAMS, set_seed
from utils import LABEL_COL, flowcap_paths


set_seed()

flowcap_ds = 'NDD'  # NDD Lymph GvHD StemCell CFSE
fcs_path, labels_path = flowcap_paths(flowcap_ds, '/mnt/data-ssd/cyto/flowcap1')
label_col = LABEL_COL

df = get_files_df(fcs_path, labels_path)
#df = df[['fcs']]
df = df.iloc[:2]
vocab = unique_labels(labels_path, label_col)
ds_kwargs = {'vocab': vocab, 'multiclass_col': label_col}

model_path = f'/home/lfisch/Projects/gatenet/data/models/flowcap/{flowcap_ds}.pth'
result = predict(model_path, df, HPARAMS, ds_kwargs=ds_kwargs)
