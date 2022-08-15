from fastai.basics import nn, np, torch, random
from sklearn.model_selection import KFold


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_kfolds(n: int, folds: int) -> list:
    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    kfolds = []
    for train_valid_split in kf.split(np.arange(n)):
        kfolds.append(train_valid_split)
    return kfolds


def weight_reset(m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d)):
        m.reset_parameters()
