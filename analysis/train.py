from fastai.basics import torch, ClassificationInterpretation

from src.data.files import get_files_df
from src.data.utils import unique_labels
from src.train import train
from utils import LABEL_COL, SCALING, flowcap_paths


def train_flowcap(hparams, flowcap_ds, flowcap_path, model_path, label_col=LABEL_COL):
    fcs_path, labels_path = flowcap_paths(flowcap_ds, flowcap_path)

    df = get_files_df(fcs_path, labels_path)
    vocab = unique_labels(labels_path, label_col)
    ds_kwargs = {'vocab': vocab, 'multiclass_col': label_col, 'fcs_scaling': SCALING[flowcap_ds]}

    learner = train(df.iloc[:8], df.iloc[-8:], hparams, ds_kwargs=ds_kwargs)

    model_path = f'{model_path}/flowcap/{flowcap_ds}.pth'
    torch.save(learner.model.state_dict(), model_path)


if __name__ == '__main__':
    from src.utils import HPARAMS, set_seed

    set_seed()

    flowcap_dir = '/mnt/data-ssd/cyto/flowcap1'
    ds = 'NDD'  # NDD Lymph GvHD StemCell CFSE

    hparams = HPARAMS
    hparams.update({'epochs': 10})
    model_dir = '/home/lfisch/Projects/gatenet/data/models'

    train_flowcap(hparams, ds, flowcap_dir, model_dir)
