from pandas import DataFrame
from fastai.basics import np, math, patch, L, Learner, DataLoaders
from fastai.callback.all import combined_cos, Callback, ParamScheduler, CancelTrainException

from .data.set import GatingDataset
from .data.loader import GateNetLoader
from .metrics import METRICS
from .model import GateNet
from .loss import get_weighted_loss
HPARAMS = {'iters': 5000, 'iters_min': 50, 'epochs_max': 10, 'lr': .002, 'bs': 1024, 'gamma': 5, 'beta': .99,
           'n_filter': (1024, 512, 256), 'n_filter_context': (64, 48), 'n_head_nodes': (32,),
           'balance_ids': False, 'oversample_beta': .999, 'n_context_events': 1000}
DL_KEYS = ('bs', 'iters_min', 'epochs_max', 'n_context_events', 'balance_ids', 'oversample_beta')


def train(df_train: DataFrame, df_valid: DataFrame, hparams: dict,
          ds_kwargs: dict = None, dl_kwargs: dict = None) -> Learner:
    if df_valid is None:
        df_valid = df_train.copy()
    dl_kwargs.update(**{k: hparams[k] for k in DL_KEYS})
    dls = get_dls(df_train, df_valid, ds_kwargs, dl_kwargs)
    hparams.update({'n_markers': dls.dataset.fcs.shape[1], 'n_gates': len(dls.dataset.vocab)})
    learn = get_learner(dls, hparams)
    if 'iters' in hparams:
        learn.fit_one_cycle_iterations(hparams['iters'], hparams['iters_min'], hparams['epochs_max'], hparams['lr'])
    else:
        learn.fit_one_cycle(hparams['epochs_max'], hparams['lr'])
    return learn


def get_dls(df_train: DataFrame, df_valid: DataFrame, ds_kwargs: dict, dl_kwargs: dict) -> DataLoaders:
    if ds_kwargs is None:
        ds_kwargs = {}
    ds_valid_kwargs = {**ds_kwargs, 'keep_events_labels': None, 'max_events': None}
    ds_train, ds_valid = GatingDataset(df_train, **ds_kwargs), GatingDataset(df_valid, **ds_valid_kwargs)
    epoch_iters = math.ceil(len(ds_train) / dl_kwargs['bs'])
    if dl_kwargs['epochs_max'] * epoch_iters < dl_kwargs['iters_min']:
        dl_kwargs.update({'bs': math.floor(dl_kwargs['epochs_max'] * len(ds_train) / dl_kwargs['iters_min'])})
    dl_train = GateNetLoader(ds_train, **dl_kwargs, shuffle=True, drop_last=False)
    dl_valid_kwargs = {**dl_kwargs, 'balance_ids': False, 'oversample_beta': 0}
    dl_valid = GateNetLoader(ds_valid, **dl_valid_kwargs, shuffle=False, drop_last=False)
    return DataLoaders(dl_train, dl_valid)


def get_learner(dls: DataLoaders, hparams: dict, metrics: dict = METRICS):
    model = GateNet(hparams)
    loss = get_weighted_loss(dls, hparams['beta'], hparams['gamma'])
    cb = StopTraining(hparams['iters'])
    return Learner(dls.cuda(), model.cuda(), loss_func=loss, metrics=list(metrics.values()), cbs=[cb]).to_fp16()


@patch
def fit_one_cycle_iterations(self: Learner, iters, iters_min=50, epochs_max=None, lr_max=None, div=25.,
                             div_final=1e5, pct_start=0.25, wd=None, moms=None, cbs=None, reset_opt=False):
    if self.opt is None:
        self.create_opt()
    self.opt.set_hyper('lr', self.lr if lr_max is None else lr_max)
    lr_max = np.array([h['lr'] for h in self.opt.hypers])
    scheds = {'lr': combined_cos(pct_start, lr_max/div, lr_max, lr_max/div_final),
              'mom': combined_cos(pct_start, *(self.moms if moms is None else moms))}
    epochs = math.ceil(iters / len(self.dls.train))
    if epochs > epochs_max or epochs_max * len(self.dls.train) < iters_min:
        if epochs_max * len(self.dls.train) < iters_min:
            epochs = math.ceil(iters_min / len(self.dls.train))
        else:
            epochs = min(epochs, epochs_max)
        self.fit(epochs, cbs=ParamScheduler(scheds) + L(cbs), reset_opt=reset_opt, wd=wd)
    else:
        speedup = (epochs * len(self.dls.train)) / iters
        lr = lambda p: scheds['lr'](speedup * p)
        mom = lambda pct: scheds['mom'](speedup * pct)
        sched = {'lr': lr, 'mom': mom}
        self.fit(epochs, cbs=ParamScheduler(sched)+L(cbs), reset_opt=reset_opt, wd=wd)


class StopTraining(Callback):
    order = 0

    def __init__(self, iterations=6000):
        super().__init__()
        self.iterations = iterations

    def after_step(self):
        if self.train_iter > self.iterations:
            raise CancelTrainException()
