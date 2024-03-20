from fastai.basics import pd, torch

from src.train import train
from src.utils import HPARAMS, set_seed
from utils import FCS_COL_NAMES, PEDIGREES, NO_REST_PANELS, get_ms_files_df


def train_total_ms(dirs: dict, idxs: list, probe: str, pedigree: tuple, hparams: dict = HPARAMS,
                   fcs_col_names: list = FCS_COL_NAMES, no_rest_panel: list = NO_REST_PANELS):
    means = pd.read_csv(dirs['mean_stats'], index_col=0)
    stds = pd.read_csv(dirs['std_stats'], index_col=0)
    for parent, kids in pedigree:
        df = get_ms_files_df(dirs, idxs, probe, panel := '|'.join(kids))
        print(f'TRAIN {panel}')

        vocab = kids if panel in no_rest_panel else kids + ['rest']
        ds_kwargs = {'pre_gate': parent, 'vocab': vocab, 'fcs_col_names': fcs_col_names, 'fcs_cols': FCS_COL_NAMES[:-1],
                     'max_events': 10000}  # , 'keep_events_labels': ['Plasma cells'] if 'Plasma cells' in vocab else None}
        dl_kwargs = {'fcs_norm_stats': {c: (means.loc[panel, c], stds.loc[panel, c]) for c in stds.columns}}
        learner = train(df, df.iloc[90:100], hparams, ds_kwargs=ds_kwargs, dl_kwargs=dl_kwargs)
        torch.save(learner.model.state_dict(), f'{dirs["model"]}/{panel}.pth')


if __name__ == '__main__':
    set_seed()
    path = '/mnt/data-ssd/cyto/fastcyto_dirs/extracted_stuff_big'
    idxs = None  # None [i for i in range(100)]
    probe = 'B'
    pedi_name = 'total'  # small total
    pedi = PEDIGREES[pedi_name]

    # pedi = (('CD45+', ['Granulos', 'Monos', 'Lymphos']), )

    paths = {'fcs': f'{path}/lmds', 'labels': f'{path}/labels', 'gates': f'{path}/labels',
             'model': f'/home/lfisch/Projects/gatenet/data/models/ms/pedigree_{pedi_name}/{probe}',
             # 'model': f'/home/lfisch/Projects/gatenet/data/models/ms/test/{probe}',
             'mean_stats': f'{path}/means_{probe}.csv', 'std_stats': f'{path}/stds_{probe}.csv'}

    hparams = HPARAMS
    # hparams.update({'iterations': 300})
    print(probe, pedi_name)
    print(hparams)

    train_total_ms(paths, idxs, probe, pedi, hparams)
