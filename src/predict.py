from typing import Iterable
from pandas import DataFrame
from fastai.basics import pd, torch, F, Learner

from .train import get_dls, get_learner
from .data.scaling import apply_scaling, INV_SCALING_METHODS


def predict(model_path: str, df: DataFrame, hparams: dict, ds_kwargs: dict = None,
            outputs: Iterable = ('data', 'preds', 'targs')) -> DataFrame:
    dls = get_dls(df, df, hparams['bs'], hparams['n_context_events'], ds_kwargs)
    if len(dls.fcs) == 0:
        return pd.DataFrame([], columns=ds_kwargs['vocab'] + ['id'])
    hparams.update({'n_markers': len(dls.dataset.fcs.columns) - 1, 'n_gates': len(dls.dataset.vocab)})
    learn = get_learner(dls, hparams)
    checkpoint = torch.load(model_path)
    learn.model.load_state_dict(checkpoint)
    result = get_learner_results(learn, outputs)
    return result


def get_learner_results(learn: Learner, keep_outputs: Iterable = ('data', 'preds', 'targs')) -> DataFrame:
    markers = learn.dls.fcs.columns.tolist()[:-1]
    gates = list(learn.dls.vocab)
    act = None if len(gates) == 2 else lambda x: x
    total_output = learn.get_preds(with_input=True, reorder=False, act=act)
    output_map = {'data': 0, 'preds': 1, 'targs': 2}
    output_dfs = []
    for ko in keep_outputs:
        output_tensor = total_output[output_map[ko]]
        df = output_tensor_to_df(output_tensor, ko, markers, gates, learn.dls.valid_ds.fcs.index)
        output_dfs.append(df)
    df = pd.concat(output_dfs + [learn.dls.valid_ds.fcs.iloc[:, -1:]], axis=1)
    if 'data' in keep_outputs:
        df.loc[:, markers] = apply_scaling(df.loc[:, markers], learn.dls.fcs_scaling, INV_SCALING_METHODS)
    return df


def output_tensor_to_df(output, name, markers, gates, idx):
    output_cols = {'data': markers, 'preds': [f'{g}_pred' for g in gates], 'targs': gates}
    if name == 'data':
        output = output[0]
    elif name == 'targs' or (name == 'preds' and len(gates) == 2):
        output = to_one_hot_encoded(output.squeeze(), len(gates))
    return DataFrame(output, idx, output_cols[name])


def to_one_hot_encoded(x: torch.Tensor, n_classes: int) -> torch.Tensor:
    if n_classes > 2:
        return F.one_hot(x, n_classes)
    else:
        return torch.stack([1 - x, x], 1)
