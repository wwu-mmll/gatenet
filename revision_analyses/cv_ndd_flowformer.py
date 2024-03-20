from fastai.basics import pd, np, glob
from src.data.files import get_files_df
from src.data.utils import unique_labels
from src.eval import evaluate_cv
from src.utils import set_seed

from revision_analyses.cv_ms_flowformer import cv


set_seed()

flowcap_dir = '/mnt/data-ssd/cyto/flowcap1'
ds = 'NDD'

folds = np.load('/home/lfisch/Projects/gatenet_old/paper_analyses/folds_ndd.npy', allow_pickle=True)

fcs_dir = f'{flowcap_dir}/Data/FCM/fcs/{ds}/FCS'

all_events = pd.concat([pd.read_csv(f) for f in glob.glob(f'{flowcap_dir}/Data/FCM/csv/NDD/CSV/*.csv')])
means = all_events.mean()
stds = all_events.std()
means.to_csv('/home/lfisch/Projects/gatenet_old/revision_analyses/ndd_mean.csv')
stds.to_csv('/home/lfisch/Projects/gatenet_old/revision_analyses/ndd_std.csv')

label_dir = f'{flowcap_dir}/Data/Labels/{ds}'

df = get_files_df(fcs_dir, label_dir)
vocab = unique_labels(label_dir, label_col := 0)

ds_kwargs = {'vocab': vocab, 'multiclass_col': 0, 'fcs_means': means, 'fcs_stds': stds}

result = cv(df, ds_kwargs, folds)
result.to_csv('/home/lfisch/Projects/gatenet_old/data/results/ms/flowformer_ndd_results_backup.csv')
print(result)
