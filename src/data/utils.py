import fcsparser
from typing import Union, Iterable
from fastai.data.all import pd, glob
FCS_SUFFIXES = ('.fcs', '.lmd', '.LMD')


def fcs_label_df(unique_substr, fcs_dir, labels_dir):
    fcs_filepath = filepaths(fcs_dir, unique_substr, list(FCS_SUFFIXES))[0]
    label_filepath = filepaths(labels_dir, unique_substr, '.csv')[0]
    fcs = fcsparser.parse(fcs_filepath, reformat_meta=True)[1]
    label = pd.read_csv(label_filepath)
    return pd.concat([fcs, label], axis=1)


def unique_labels(label_dir: str, label_col: Union[int, str] = 0):
    possible_labels = []
    for f in filepaths(label_dir):
        df = pd.read_csv(f)
        labels = df.iloc[:, label_col] if isinstance(label_col, int) else df.loc[:, label_col]
        possible_labels += labels.unique().tolist()
    labels = list(set(possible_labels))
    return [str(label) for label in labels]


def filepaths(path: str, substr: Union[list, str] = None, file_types: Union[list, str] = None):
    substr = '' if substr is None else substr
    substr = '*'.join(substr) if isinstance(substr, list) else substr
    file_types = '' if file_types is None else file_types
    file_types = [file_types] if isinstance(file_types, str) else file_types
    filepaths = []
    for ft in file_types:
        filepaths += sorted(glob.glob(f'{path}/*{substr}*{ft}'))
    return pd.Series(sorted(filepaths)).reset_index(drop=True)
