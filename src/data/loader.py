from typing import Tuple
from fastai.data.all import np, pd, math, torch, tensor, itertools, DataLoader
from fastcore.basics import Inf

from .set import GatingDataset


class GateNetLoader(DataLoader):
    def __init__(self, dataset: GatingDataset, bs: int = 500, n_context_events: int = 500, balance_ids: bool = False,
                 oversample_beta: float = 0, fcs_scaling: dict = None, fcs_norm_stats: dict = None, **kwargs):
        super().__init__(dataset, bs, **kwargs)
        self.n_context_events = n_context_events
        self.dataset.bs = bs
        self.balance_ids = balance_ids
        self.beta = oversample_beta
        self.id_codes = self.ids.cat.codes.values

        fcs_scaling = {'forward': {}, 'backward': {}} if fcs_scaling is None else fcs_scaling
        self.fcs_scaling = {m: fcs_scaling['all'] for m in self.fcs.columns} if 'all' in fcs_scaling else fcs_scaling
        fcs_norm_stats = {m: (0, 1) for m in self.fcs.columns} if fcs_norm_stats is None else fcs_norm_stats
        self.fcs_norm_stats = {m: fcs_norm_stats['all'] for m in self.fcs.columns} if 'all' in fcs_norm_stats else fcs_norm_stats
        self.fcs = self.encode(self.fcs)
        if len(self.fcs) != 0:
            self.fcs_context_tensor = self.get_fcs_context_tensor(self.fcs, self.id_codes, memory_limit=100000)
        else:
            self.fcs_context_tensor = None

    def encode(self, fcs):
        fcs = self.scale(fcs, self.fcs_scaling['forward'])
        fcs = self.normalize(fcs, self.fcs_norm_stats)
        return fcs

    def decode(self, fcs):
        fcs = self.denormalize(fcs, self.fcs_norm_stats)
        fcs = self.scale(fcs, self.fcs_scaling['backward'])
        return fcs

    def get_idxs(self):
        idxs = Inf.count if self.indexed else Inf.nones
        if self.n is not None:
            idxs = list(itertools.islice(idxs, self.n))
        idx_weights = []
        if self.balance_ids:
            _, counts = np.unique(self.id_codes, return_counts=True)
            idx_weights.append(np.concatenate([np.ones(c) / c for c in counts]))
        if self.beta > 0.:
            labels = self.labels.copy()
            unique_labels, counts = np.unique(labels, return_counts=True)
            weights = {label: (1 - self.beta) / (1 - self.beta ** count) for label, count in zip(unique_labels, counts)}
            idx_weights.append(labels.cat.codes.map(weights).values)
        if idx_weights:
            p = np.ones(len(idxs))
            for w in idx_weights:
                p *= w
            p /= p.sum()
            idxs = np.random.choice(np.array(idxs), self.n, p=p / p.sum())
            idxs = np.sort(idxs).tolist()
        if self.shuffle:
            idxs = self.shuffle_fn(idxs)
        return idxs

    def create_item(self, s: int): return s

    def create_batch(self, b: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context_events = self.create_context_events(b)
        x, y = self.dataset[b]
        x = tensor(x, dtype=torch.bfloat16)
        y = tensor(y)
        return x, context_events, y

    def create_context_events(self, b: list) -> torch.Tensor:
        ids = self.id_codes[b].tolist()
        rand_idxs = torch.randint(self.fcs_context_tensor.shape[1], size=(self.n_context_events, ))
        return self.fcs_context_tensor[:, rand_idxs, :][ids]

    @staticmethod
    def scale(fcs, scaling):
        for col, func in scaling.items():
            fcs[col] = func(fcs[col])
        return fcs

    @staticmethod
    def normalize(fcs, stats):
        fcs = fcs.sub([stats[marker][0] for marker in fcs.columns])
        fcs = fcs.div([stats[marker][1] for marker in fcs.columns])
        return fcs

    @staticmethod
    def denormalize(fcs, stats):
        fcs = fcs.mul([stats[marker][1] for marker in fcs.columns])
        fcs = fcs.add([stats[marker][0] for marker in fcs.columns])
        return fcs

    @staticmethod
    def get_fcs_context_tensor(fcs: pd.DataFrame, id_codes, memory_limit=100000) -> torch.Tensor:
        context = []
        fcs_tensor = tensor(fcs.values, dtype=torch.bfloat16, device='cuda')
        unique_ids, id_event_counts = np.unique(id_codes, return_counts=True)
        max_event_count = id_event_counts.max().item()
        n_id_events = min(max_event_count, memory_limit)
        for id_ in unique_ids:
            id_events = fcs_tensor[id_codes == id_, :]
            if id_events.shape[0] == 0:
                id_events = torch.zeros((n_id_events, fcs_tensor.shape[1]), dtype=torch.bfloat16, device='cuda')
            elif id_events.shape[0] != 0 and id_events.shape[0] < n_id_events:
                id_events = id_events.repeat(math.ceil(n_id_events / id_events.shape[0]), 1)
            context.append(id_events[:n_id_events])
        return torch.stack(context)
