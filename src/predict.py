from fastai.basics import pd, torch, F, Learner

from .train import get_dls, get_learner, DL_KEYS


def predict(model_path: str, df: pd.DataFrame, hparams: dict, ds_kwargs: dict = None, dl_kwargs: dict = None,
            with_data: bool = True, with_targs: bool = True, progress_bar: bool = False) -> pd.DataFrame:
    dl_kwargs.update(**{k: hparams[k] for k in DL_KEYS})
    dls = get_dls(df.iloc[:1], df, ds_kwargs, dl_kwargs)
    if len(dls.fcs) == 0:
        return pd.DataFrame([], columns=ds_kwargs['vocab'] + ['id'])
    hparams.update({'n_markers': dls.fcs.shape[1], 'n_gates': len(dls.vocab)})
    learn = get_learner(dls, hparams)
    learn.model.load_state_dict(torch.load(model_path))
    result = get_results(learn, with_data, with_targs, progress_bar)
    return result


def get_results(learn: Learner, with_data: bool = True, with_targs: bool = True,
                progress_bar: bool = False) -> pd.DataFrame:
    marker, idx, gates = learn.dls.fcs.columns, learn.dls[1].fcs.index, list(learn.dls.vocab)
    results = []
    if with_data:
        results.append(learn.dls[1].dataset.fcs)
    if progress_bar:
        preds, targs = learn.get_preds(dl=learn.dls[1], reorder=False, act=lambda x: x)
    else:
        with learn.no_bar():
            preds, targs = learn.get_preds(dl=learn.dls[1], reorder=False, act=lambda x: x)
    results.append(pd.DataFrame(preds, idx, [f'{g}_pred' for g in gates]))
    if with_targs:
        results.append(pd.DataFrame(F.one_hot(targs.view(-1), len(gates)), idx, gates))
    results = pd.concat(results, axis=1)
    results['id'] = learn.dls[1].ids
    return results
