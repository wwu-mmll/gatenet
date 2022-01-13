from fastai.basics import pd

from src.train import train
from src.predict import get_learner_results
from src.utils import get_kfolds, weight_reset


def cv(df: pd.DataFrame, hparams: dict, ds_kwargs: dict = None, folds: int = 5,
       outputs: list = ('data', 'preds', 'targs')) -> pd.DataFrame:
    folds = get_kfolds(len(df), folds)
    cv_result = []
    for i, (idxs_t, idxs_v) in enumerate(folds):
        df_train, df_valid = df.iloc[idxs_t], df.iloc[idxs_v]
        learner = train(df_train, df_valid, hparams, ds_kwargs)
        fold_result = get_learner_results(learner, outputs)
        fold_result['fold'] = i
        cv_result.append(fold_result)
        learner.model = weight_reset(learner.model)
    return pd.concat(cv_result)
