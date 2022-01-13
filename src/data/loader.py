from typing import Tuple
from fastai.data.all import np, pd, math, torch, tensor, DataLoader

from .set import GatingDataset


class GateNetLoader(DataLoader):
    def __init__(self, dataset: GatingDataset, bs: int = 500, n_context_events: int = 500, **kwargs):
        super().__init__(dataset, bs, **kwargs)
        self.n_context_events = n_context_events
        self.dataset.bs = bs
        self.ids = self.get_ids(self.fcs)
        self.fcs_context_tensor = self.get_fcs_context_tensor(self.fcs, self.ids) if len(self.fcs) != 0 else None

    def create_item(self, s: int): return s

    def create_batch(self, b: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context_events = self.create_context_events(b)
        x, y = self.dataset[b]
        x = tensor(x)
        y = tensor(y)
        return x, context_events, y

    def create_context_events(self, b: list) -> torch.Tensor:
        ids = self.ids[b].tolist()
        rand_idxs = torch.randint(self.fcs_context_tensor.shape[1], size=(self.n_context_events, ))
        return self.fcs_context_tensor[:, rand_idxs, :][ids]

    @staticmethod
    def get_ids(fcs: pd.DataFrame) -> torch.Tensor:
        return fcs.id.astype('category').cat.codes.values

    @staticmethod
    def get_fcs_context_tensor(fcs: pd.DataFrame, ids) -> torch.Tensor:
        context = []
        fcs_tensor = torch.from_numpy(fcs.iloc[:, :-1].values).type(torch.cuda.HalfTensor)
        id_event_counts = np.bincount(ids)
        max_event_count = id_event_counts.max().item()
        memory_limit = 100000
        n_id_events = min(max_event_count, memory_limit)
        for id in np.unique(ids):
            id_events = fcs_tensor[ids == id, :]
            if id_events.shape[0] == 0:
                id_events = torch.zeros((n_id_events, fcs_tensor.shape[1]), dtype=torch.cuda.HalfTensor)
            elif id_events.shape[0] != 0 and id_events.shape[0] < n_id_events:
                id_events = id_events.repeat(math.ceil(n_id_events / id_events.shape[0]), 1)
            context.append(id_events[:n_id_events])
        return torch.stack(context)
