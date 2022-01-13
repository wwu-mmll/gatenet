from src.data.files import get_files_df
from src.data.utils import unique_labels
from src.cv import cv
from src.eval import evaluate_cv
from utils import LABEL_COL, flowcap_paths


def cv_flowcap(n_folds, hparams, flowcap_ds, flowcap_path, result_path, label_col=LABEL_COL):
    fcs_path, labels_path = flowcap_paths(flowcap_ds, flowcap_path)

    df = get_files_df(fcs_path, labels_path)
    vocab = unique_labels(labels_path, label_col)
    ds_kwargs = {'vocab': vocab, 'multiclass_col': label_col}

    result = cv(df, hparams, folds=n_folds, ds_kwargs=ds_kwargs)
    evaluate_cv(result, result_path)


if __name__ == '__main__':
    from src.utils import HPARAMS, set_seed

    set_seed()

    flowcap_dir = '/mnt/data-ssd/cyto/flowcap1'
    datasets = ['GvHD', 'NDD', 'Lymph', 'StemCell', 'CFSE']

    hparams = HPARAMS
    hparams.update({'iterations': 60000})
    num_folds = 10

    for ds in datasets:
        result_dir = f'/home/lfisch/Projects/gatenet/data/results/flowcap/{ds}'
        cv_flowcap(num_folds, hparams, ds, flowcap_dir, result_dir)
