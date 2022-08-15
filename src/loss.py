from typing import Callable
from fastai.basics import torch, FocalLossFlat, DataLoaders


def get_weighted_loss(dls: DataLoaders, beta: float = .99, gamma: float = 5.) -> Callable:
    cls_counts = dls.train_ds.labels.value_counts().sort_index()
    w = (1 - beta) / (1 - beta ** cls_counts.values)
    w = torch.tensor(w, dtype=torch.float32).cuda()
    w /= w.mean()
    return FocalLossFlat(weight=w, gamma=gamma)
