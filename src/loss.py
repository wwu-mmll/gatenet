from typing import Callable
from fastai.basics import torch, CrossEntropyLossFlat, BCEWithLogitsLossFlat, DataLoaders


def get_weighted_loss(dls: DataLoaders, focal_loss: bool = True) -> Callable:
    cls_counts = dls.train_ds.labels.value_counts().sort_index()
    w = torch.tensor([max(cls_counts) / n if n > 0 else max(cls_counts) for n in cls_counts]).cuda()
    if len(dls.dataset.vocab) == 2:
        return BCEWithLogitsLossFlat()
    else:
        return FocalLossFlat(weight=w) if focal_loss else CrossEntropyLossFlat(weight=w)


# TODO: Build newer version fastai env which includes this loss
class FocalLossFlat(CrossEntropyLossFlat):
    """
    Same as CrossEntropyLossFlat but with focal paramter, `gamma`. Focal loss is introduced by Lin et al.
    https://arxiv.org/pdf/1708.02002.pdf. Note the class weighting factor in the paper, alpha, can be
    implemented through pytorch `weight` argument in nn.CrossEntropyLoss.
    """
    y_int = True
    #@use_kwargs_dict(keep=True, weight=None, ignore_index=-100, reduction='mean')
    def __init__(self, *args, gamma=2, axis=-1, **kwargs):
        self.gamma = gamma
        self.reduce = kwargs.pop('reduction') if 'reduction' in kwargs else 'mean'
        super().__init__(*args, reduction='none', axis=axis, **kwargs)
    def __call__(self, inp, targ, **kwargs):
        ce_loss = super().__call__(inp, targ, **kwargs)
        pt = torch.exp(-ce_loss)
        fl_loss = (1-pt)**self.gamma * ce_loss
        return fl_loss.mean() if self.reduce == 'mean' else fl_loss.sum() if self.reduce == 'sum' else fl_loss