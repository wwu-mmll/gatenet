import seaborn as sn
from typing import Tuple
from pandas import DataFrame
from fastai.basics import np, torch, tensor, Path
from sklearn.metrics import confusion_matrix, classification_report

from .metrics import get_metrics


def evaluate_cv(res: DataFrame, save_dir: str = None) -> Tuple[DataFrame, DataFrame, DataFrame]:
    filenames = ('metrics', 'confusion_matrix', 'report')
    metrics = cv_metrics_table(res)
    cm = conf_matrix(res)
    report = clf_report(res)
    if save_dir is not None:
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        [df.to_csv(f'{save_dir}/{f}.csv') for df, f in zip([metrics, cm, report], filenames)]
    return metrics, cm, report


def cv_metrics_table(res: DataFrame) -> DataFrame:
    preds_cols, targs_cols = get_preds_targs_cols(res)
    metrics = get_metrics(n_classes=len(targs_cols), sigmoid=False)
    table = []
    for i in res['fold'].unique():
        df_fold = res[res.fold == i]
        preds, targs = df_fold[preds_cols], df_fold[targs_cols]
        metrics_fold = calc_metrics(preds, targs, metrics)
        table.append(metrics_fold)
    table = DataFrame(table)
    table.loc['mean'] = table.mean()
    table.loc['std'] = table.std()
    return table


# TODO: Can this be done cleaner?
def calc_metrics(preds: DataFrame, targs: DataFrame, metrics: dict) -> dict:
    binary = preds.shape[1] == 2
    preds, targs = tensor(preds), tensor(targs)
    targs = torch.argmax(targs, dim=1)
    preds_ = torch.argmax(preds, dim=1)
    values = [m(preds, targs) if not binary and name == 'ACC' else m(preds_, targs) for name, m in metrics.items()]
    values = [m if isinstance(m, float) else m.item() for m in values]
    return {name: value for (name, _), value in zip(metrics.items(), values)}


def plot_confusion_matrix(res: DataFrame):
    cm = conf_matrix(res)
    ax = sn.heatmap(cm, annot=True, fmt='g')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')


def conf_matrix(res: DataFrame) -> DataFrame:
    preds, targs = get_preds_targs(res)
    cm_values = confusion_matrix(np.argmax(targs, axis=1), np.argmax(preds, axis=1))
    return DataFrame(cm_values)


def clf_report(res: DataFrame) -> DataFrame:
    preds, targs = get_preds_targs(res)
    report_data = classification_report(np.argmax(targs, axis=1), np.argmax(preds, axis=1), output_dict=True)
    report = DataFrame.from_dict(report_data)
    return report.T


def get_preds_targs(res: DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    preds_cols, targs_cols = get_preds_targs_cols(res)
    return res[preds_cols].values, res[targs_cols].values


def get_preds_targs_cols(res: DataFrame) -> Tuple[list, list]:
    preds_cols = [c for c in res.columns if '_pred' in c]
    targs_cols = [c.replace('_pred', '') for c in preds_cols]
    return preds_cols, targs_cols
