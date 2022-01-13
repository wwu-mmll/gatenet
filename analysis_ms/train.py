import torch

from src.train import train
from src.utils import HPARAMS, set_seed
from utils import FCS_COL_NAMES, PEDIGREES, NO_REST_PANELS, get_ms_files_df


def train_total_ms(dirs: dict, idxs: list, probe: str, pedigree: tuple, fcs_col_names: list = FCS_COL_NAMES,
                   hparams: dict = HPARAMS, no_rest_panel: list = NO_REST_PANELS):
    for parent, kids in pedigree:
        df = get_ms_files_df(dirs, idxs, probe, gate_panel := '_'.join(kids))
        print(f'TRAIN {gate_panel}')

        vocab = kids if parent in no_rest_panel else kids + ['rest']
        ds_kwargs = {'pre_gate': parent, 'vocab': vocab, 'fcs_col_names': fcs_col_names}

        learner = train(df, df.iloc[5:10], hparams, ds_kwargs=ds_kwargs)
        torch.save(learner.model.state_dict(), f'{dirs["model"]}/{gate_panel}.pth')


if __name__ == '__main__':
    set_seed()
    path = '/mnt/data-ssd/cyto/fastcyto_dirs/extracted_stuff_big'
    idxs = [i for i in range(10)]  # None [i for i in range(10)]
    probe = 'B'
    pedi_name = 'total'  # small total
    pedi = PEDIGREES[pedi_name]

    paths = {'fcs': f'{path}/lmds', 'labels': f'{path}/labels', 'gates': f'{path}/labels',
             #'model': f'/home/lfisch/Projects/gatenet/data/models/ms/pedigree_{pedi_name}/{probe}',
             'model': f'/home/lfisch/Projects/gatenet/data/models/ms/test/{probe}'}

    hparams = HPARAMS
    hparams.update({'iterations': 60000})
    print(probe, pedi_name)
    print(hparams)

    train_total_ms(paths, idxs, probe, pedi)
