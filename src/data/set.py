import warnings
import fcsparser
from typing import Tuple, Union
from pandas import DataFrame, Series
from fastai.data.all import np, pd, Path, Categorize, CategoryMap

from .scaling import apply_scaling


class GatingDataset:
    def __init__(self, df: DataFrame, pre_gate: str = None, vocab: list = None, fcs_col_names: list = None,
                 fcs_scaling: dict = None, multiclass_col: Union[str, int] = None):
        """
        Dataset of cytometry events with respective gating labels
        :param df: DataFrame containing the filepaths. Columns must be named 'fcs', 'labels' and optionally 'gates'
        If there are no labels there should be no 'labels' column. Dummy labels will be generated
        Additional context fcs-Files can be loaded, using column names f'additional_context_fcs_{i}' with i = 0, 1, ...
        :param pre_gate: Gate column which will be used to filter events
        :param vocab: Label columns (or label values if multiclass_col is not None) which will be used as targets.
        If the last element in vocab is 'rest', a 'rest'-class (i.e. column) will be inferred
        :param fcs_col_names: FCS data columns will be renamed (i.e. replace) by these names
        :param fcs_scaling: Scaling methods with the respective FCS data columns they should be applied to
        :param multiclass_col: Label column containing multiple classes. If None, labels should be one-hot encoded
        """
        self.fcs_scaling = fcs_scaling
        self.vocab = Categorize(vocab=vocab).vocab
        self.fcs = self.load_fcs(df, fcs_col_names)
        gate = None if pre_gate is None else self.load_pre_gate(df, pre_gate)
        self.fcs = self.fcs if pre_gate is None else self.fcs[gate]
        self.labels = self.load_labels(df, multiclass_col, gate) if 'labels' in df.columns else self.dummy_labels()

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.fcs.iloc[idx, :-1].values, self.labels.iloc[idx].values

    def __len__(self) -> int:
        return len(self.fcs)

    def load_fcs(self, df: DataFrame, col_names: list) -> DataFrame:
        fcs = self.load_files(df.fcs, add_id_col=True, col_names=col_names)
        if self.fcs_scaling is None:
            self.fcs_scaling = {'arcsinh': fcs.columns[:-1]}
        fcs = apply_scaling(fcs, self.fcs_scaling)
        return fcs

    def load_labels(self, df: DataFrame, multiclass_col: Union[str, int], gate: Series = None) -> Series:
        labels = self.load_files(df.labels, add_id_col=False)
        labels = labels if gate is None else labels[gate]
        if multiclass_col is None:
            labels = labels[[v for v in self.vocab if v != 'rest']]
            if 'rest' in self.vocab:
                labels = self.add_rest_class(labels)
            labels = self.reverse_one_hot_encode(labels, self.vocab)
        else:
            labels = labels[multiclass_col]
        return self.to_categorical(labels, self.vocab)

    def dummy_labels(self):
        labels = Series(np.zeros(len(self.fcs), dtype=int), index=self.fcs.index)
        # labels.iloc[-len(self.vocab):] = np.arange(len(self.vocab))  # to avoid torch warning
        return self.to_categorical(labels, self.vocab)

    def load_pre_gate(self, df: DataFrame, pre_gate: str):
        gates = self.load_files(df.gates, add_id_col=False)
        return gates[pre_gate] > .5

    def load_files(self, fpaths: Series, add_id_col: bool, col_names: list = None) -> DataFrame:
        files = []
        for f in fpaths:
            file = self.load_file(f)
            if col_names is not None:
                file.columns = col_names
            files.append(file)
        merged_files = pd.concat(files).reset_index(drop=True)
        if add_id_col:
            merged_files['id'] = np.concatenate([[Path(f).stem] * len(file) for f, file in zip(fpaths, files)]).flat
        return merged_files

    def load_file(self, filepath: str) -> DataFrame:
        if Path(filepath).suffix in ('.fcs', '.lmd', '.LMD'):
            load_func = self.load_fcs_file
        elif Path(filepath).suffix == '.csv':
            load_func = self.load_csv_file
        else:
            raise ValueError(f'File format of {filepath} is not supported by dataset')
        return load_func(filepath)

    def add_rest_class(self, labels):
        rest = labels.eq(0).all(1).astype(int)
        if rest.sum() > 0:
            labels['rest'] = rest
        else:
            warnings.warn('No zero labels. Rest class could not be added. Will be removed in vocab', UserWarning)
            self.vocab = Categorize(vocab=[v for v in self.vocab if v != 'rest']).vocab
        return labels

    @staticmethod
    def load_fcs_file(file: str) -> DataFrame:
        return fcsparser.parse(file, reformat_meta=True)[1]

    @staticmethod
    def load_csv_file(file: str) -> DataFrame:
        return pd.read_csv(file)

    @staticmethod
    def reverse_one_hot_encode(labels: DataFrame, vocab: CategoryMap) -> Series:
        return labels.idxmax(axis=1).map(vocab.o2i)

    @staticmethod
    def to_categorical(labels: Series, vocab: CategoryMap):
        return labels.astype(pd.CategoricalDtype(categories=[i for i in range(len(vocab))]))
