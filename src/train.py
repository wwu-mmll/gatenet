from pandas import DataFrame
from fastai.basics import math, DataLoaders, Learner
from fastai.callback.all import *

from .data.set import GatingDataset
from .data.loader import GateNetLoader
from .metrics import get_metrics
from .model import GateNet
from .loss import get_weighted_loss


def train(df_train: DataFrame, df_valid: DataFrame, hparams: dict, ds_kwargs: dict = None, cbs: list = None) -> Learner:
    dls = get_dls(df_train, df_valid, hparams['bs'], hparams['n_context_events'], ds_kwargs)
    hparams.update({'n_markers': len(dls.dataset.fcs.columns) - 1, 'n_gates': len(dls.dataset.vocab)})
    if 'iterations' in hparams:
        hparams.update({'epochs': math.ceil(hparams['iterations'] / len(dls.train))})
    learn = get_learner(dls, hparams)
    if cbs is not None:
        learn.add_cbs(cbs)
    learn.fit_one_cycle(hparams['epochs'], hparams['lr'])
    return learn


def get_dls(df_train: DataFrame, df_valid: DataFrame, bs: int, n_context_events: int, ds_kwargs: dict):
    if ds_kwargs is None:
        ds_kwargs = {}
    ds_train, ds_valid = GatingDataset(df_train, **ds_kwargs), GatingDataset(df_valid, **ds_kwargs)
    dl_train = GateNetLoader(ds_train, bs, n_context_events, shuffle=True, drop_last=True)
    dl_valid = GateNetLoader(ds_valid, bs, n_context_events, shuffle=False, drop_last=True)
    return DataLoaders(dl_train, dl_valid)


def get_learner(dls: DataLoaders, hparams: dict, focal_loss: bool = True):
    model = GateNet(hparams)
    loss = get_weighted_loss(dls, focal_loss)
    metrics = get_metrics(n_classes=len(dls.dataset.vocab))
    metrics = [v for _, v in metrics.items()]
    return Learner(dls.cuda(), model.cuda(), loss_func=loss, metrics=metrics).to_fp16()
