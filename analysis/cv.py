from fastai.data.all import np, pd

from src.data.files import get_files_df
from src.data.utils import unique_labels
from src.cv import cv
from src.eval import evaluate_cv
from analysis.utils import DATASET_NAME_MAP, FCS_SCALING, FCS_STATS, LABEL_COL, flowcap_paths


def cv_flowcap(n_folds, hparams, flowcap_ds, flowcap_path, result_path, label_col=LABEL_COL):
    fcs_path, labels_path = flowcap_paths(flowcap_ds, flowcap_path)

    df = get_files_df(fcs_path, labels_path)[:6]
    vocab = unique_labels(labels_path, label_col)
    ds_kwargs = {'vocab': vocab, 'multiclass_col': label_col}
    dl_kwargs = {'fcs_norm_stats': FCS_STATS[flowcap_ds], 'fcs_scaling': FCS_SCALING[flowcap_ds]}

    result = cv(df, hparams, ds_kwargs, dl_kwargs, n_folds, with_targs=True)
    evaluate_cv(result, result_path)


def merge_flowcap_results(result_dir):
    merged_results = {}
    for ds in DATASET_NAME_MAP:
        ds_results = {}
        res_df = pd.read_csv(f'{result_dir}/{ds}/metrics.csv', index_col=[0])
        for m in res_df.columns:
            m_string = f'{res_df[m].loc["mean"]:.2f} ({res_df[m].loc["std"]:.2f})'
            ds_results.update({m: m_string})
        merged_results.update({ds: ds_results})
    return pd.DataFrame(merged_results).T


if __name__ == '__main__':
    from src.utils import HPARAMS, set_seed
    FLOWCAP_DIR = '/mnt/data-ssd/cyto/flowcap1'  # '/home/lfisch/data/cyto/flowcap1'
    RESULT_DIR = '/home/lfisch/Projects/gatenet/data/results/flowcap'

    set_seed()

    hparams = HPARAMS
    hparams.update({'iterations': 100, 'lr': .003, 'bs': 1024, 'balance_ids': True, 'oversample_beta': .999,
                    'n_context_events': 500, 'n_filter': (1024, 512, 256), 'n_filter_context': (64, 48),
                    'n_head_nodes': (32, ), 'beta': .99, 'gamma': 5.})
    num_folds = 2

    # for ds in DATASET_NAME_MAP:
    for ds in ['NDD']:
         cv_flowcap(num_folds, hparams, ds, FLOWCAP_DIR, f'{RESULT_DIR}/{ds}')

    # merged_res = merge_flowcap_results(RESULT_DIR)
    # merged_res.to_csv(f'{RESULT_DIR}/results.csv')

