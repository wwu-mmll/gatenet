from fastai.basics import pd, np

from src.train import train
from src.predict import get_results
from src.utils import get_kfolds, weight_reset


def cv(df: pd.DataFrame, hparams: dict, ds_kwargs: dict = None, dl_kwargs: dict = None, folds: [int, list] = 5,
       with_data: bool = True, with_targs: bool = False) -> pd.DataFrame:
    folds = get_kfolds(len(df), folds) if isinstance(folds, int) else folds
    cv_result = []
    for i, (idxs_t, idxs_v) in enumerate(folds):
        df_train, df_valid = df.iloc[idxs_t], df.iloc[idxs_v]
        learner = train(df_train, df_valid, hparams, ds_kwargs, dl_kwargs)
        fold_result = get_results(learner, with_data, with_targs)
        fold_result['fold'] = i
        cv_result.append(fold_result)
        learner.model = weight_reset(learner.model)
    return pd.concat(cv_result)
