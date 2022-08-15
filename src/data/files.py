from fastai.data.all import np, pd

from .utils import filepaths, FCS_SUFFIXES


def get_files_df(fcs_dir: str, label_dir: str = None, gates_dir: str = None, file_idxs: list = None,
                 substr: str = None, gate_substr: str = None) -> pd.DataFrame:
    datapaths = {}
    fcs_paths = filepaths(fcs_dir, substr, FCS_SUFFIXES)
    if file_idxs is None:
        file_idxs = np.arange(len(fcs_paths))
    datapaths.update({'fcs': fcs_paths.iloc[file_idxs]})
    if label_dir is not None:
        datapaths.update({'labels': filepaths(label_dir, substr, ['.ftr', '.csv']).iloc[file_idxs]})
    if gates_dir is not None:
        substr = substr if gate_substr is None else [substr, gate_substr]
        datapaths.update({'gates': filepaths(gates_dir, substr, ['.ftr', '.csv']).iloc[file_idxs]})
    df = pd.DataFrame(datapaths)
    return df
