import fcsparser
from typing import Tuple, Union
from fastai.data.all import np, pd, Path, warnings, Categorize, CategoryMap


class GatingDataset:
    def __init__(self, df: pd.DataFrame, pre_gate: str = None, vocab: list = None, max_events: int = None,
                 fcs_cols: list = None, fcs_col_names: list = None, multiclass_col: Union[str, int] = None,
                 keep_events_labels: list = None):
        self.vocab = Categorize(vocab=vocab).vocab
        self.file_event_counts = None
        self.fcs = self.load_fcs(df, fcs_cols, fcs_col_names)
        gate = pd.Series(True, index=self.fcs.index) if pre_gate is None else self.load_pre_gate(df, pre_gate)
        self.fcs = self.fcs[gate]
        self.labels = self.load_labels(df, multiclass_col, gate) if 'labels' in df.columns else self.dummy_labels()
        self.ids = self.load_ids(df, label_ids='labels' in df.columns, gate=gate)
        if max_events is not None:
            if isinstance(max_events, str):
                factor = int(max_events)
                median = int(self.ids.value_counts().median())
                max_events = factor * median
            gate = self.max_events_gate(max_events, keep_events_labels)
            self.fcs, self.ids, self.labels = self.fcs[gate], self.ids[gate], self.labels[gate]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.fcs.iloc[idx].values, self.labels.iloc[idx].values

    def __len__(self) -> int:
        return len(self.fcs)

    def load_ids(self, df: pd.DataFrame, label_ids: bool, gate: pd.Series = None) -> pd.Series:
        ids = [[f] * count for f, count in zip(df['labels' if label_ids else 'fcs'], self.file_event_counts)]
        ids = pd.Series(np.concatenate(ids).flat)
        ids = ids if gate is None else ids[gate]
        return ids.astype('category')

    def max_events_gate(self, max_events: int, keep_labels: list):
        keep_events = []
        for _, id_labels in self.labels.groupby(self.ids):
            id_max_events = min(len(id_labels), max_events)
            keep_events.append(id_labels.sample(id_max_events, random_state=0))
        if keep_labels is not None:
            keep_labels = [self.vocab.o2i[label] for label in keep_labels]
            keep_events.append(self.labels[self.labels.isin(keep_labels)])
        return self.labels.index.isin(pd.concat(keep_events).index)

    def load_fcs(self, df: pd.DataFrame, keep_cols: list, col_names: list) -> pd.DataFrame:
        return self.load_files(df.fcs, keep_cols=keep_cols, col_names=col_names)

    def load_labels(self, df: pd.DataFrame, multiclass_col: Union[str, int], gate: pd.Series = None) -> pd.Series:
        labels = self.load_files(df.labels)
        labels = labels if gate is None else labels[gate]
        if multiclass_col is None:
            labels = labels[[v for v in self.vocab if v != 'rest']]
            if 'rest' in self.vocab:
                labels = self.add_rest_class(labels)
            labels = self.reverse_one_hot_encode(labels, self.vocab)
        else:
            labels = labels[multiclass_col] if isinstance(multiclass_col, str) else labels.iloc[:, 0]
        return self.to_categorical(labels, self.vocab)

    def dummy_labels(self):
        labels = pd.Series(np.zeros(len(self.fcs), dtype=int), index=self.fcs.index)
        return self.to_categorical(labels, self.vocab)

    def load_pre_gate(self, df: pd.DataFrame, pre_gate: str):
        gate = self.load_files(df.gates, reset_index=False)
        gate = gate.reindex(self.fcs.index, fill_value=0) if len(gate) != len(self.fcs) else gate.reset_index(drop=True)
        return gate[pre_gate] > .5

    def load_files(self, fpaths: pd.Series, keep_cols: list = None, col_names: list = None,
                   reset_index: bool = True) -> pd.DataFrame:
        files = []
        for f in fpaths:
            file = self.load_file(f)
            file.columns = file.columns if col_names is None else col_names
            file = file if keep_cols is None else file[keep_cols]
            files.append(file)
        self.file_event_counts = [len(file) for file in files]
        files = pd.concat(files)
        return files.reset_index(drop=True) if reset_index else files

    def add_rest_class(self, labels):
        rest = labels.eq(0).all(1).astype(int)
        if rest.sum() > 0:
            labels['rest'] = rest
        else:
            warnings.warn('No zero labels. Rest class could not be added. Will be removed in vocab', UserWarning)
            self.vocab = Categorize(vocab=[v for v in self.vocab if v != 'rest']).vocab
        return labels

    @staticmethod
    def load_file(filepath: str) -> pd.DataFrame:
        if Path(filepath).suffix in ('.fcs', '.lmd', '.LMD'):
            df = fcsparser.parse(filepath, reformat_meta=True)[1]
        elif Path(filepath).suffix == '.csv':
            df = pd.read_csv(filepath)
        elif Path(filepath).suffix == '.ftr':
            df = pd.read_feather(filepath)
            df = df.set_index('index') if 'index' in df.columns else df
        else:
            raise ValueError(f'File format of {filepath} is not supported by dataset')
        return df

    @staticmethod
    def reverse_one_hot_encode(labels: pd.DataFrame, vocab: CategoryMap) -> pd.Series:
        return labels.idxmax(axis=1).map(vocab.o2i)

    @staticmethod
    def to_categorical(labels: pd.Series, vocab: CategoryMap):
        return labels.astype(pd.CategoricalDtype(categories=[i for i in range(len(vocab))]))
