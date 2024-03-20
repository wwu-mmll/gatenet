import os
import warnings
warnings.filterwarnings("ignore")
from fastai.basics import pd, Path
from tqdm import tqdm

from src.utils import set_seed, HPARAMS
from src.predict import predict
from analysis_ms.utils import PEDIGREES, FCS_COL_NAMES, NO_REST_PANELS, get_gate_panel


def predict_file(fp, dirs, pedigree, means, stds, hparams: dict = HPARAMS, fcs_col_names: list = FCS_COL_NAMES,
                 no_rest_panel: list = NO_REST_PANELS):
    fs = Path(fp).stem
    df = pd.DataFrame({'fcs': [fp]})
    for i, (parent, kids) in enumerate(pedigree):
        panel = '|'.join(kids)
        if parent is not None:
            df['gates'] = [dirs['gates'] + f'/{fs}_{get_gate_panel(parent, pedigree)}.ftr']
        ds_kwargs = {'pre_gate': parent, 'vocab': kids if panel in no_rest_panel else kids + ['rest'],
                     'fcs_col_names': fcs_col_names, 'fcs_cols': fcs_col_names[:-1]}
        dl_kwargs = {'fcs_norm_stats': {c: (means.loc[panel, c], stds.loc[panel, c]) for c in stds.columns}}
        pred = predict(dirs['model'] + f'/{panel}.pth', df, hparams, ds_kwargs, dl_kwargs, False, False)
        torch.cuda.empty_cache()
        save_pred(pred, dirs, fs, panel)
    merge_preds(dirs['preds'], fs, pedigree).to_feather(f'{dirs["preds"]}/{fs}.ftr')
    merge_preds(dirs['gates'], fs, pedigree).to_feather(f'{dirs["gates"]}/{fs}.ftr')


def save_pred(pred, dirs, fs, panel):
    pred = pred.iloc[:, :-1]
    pred = pred.rename(columns={c: c.replace('_pred', '') for c in pred.columns})
    pred.reset_index().to_feather(f'{dirs["preds"]}/{fs}_{panel}.ftr')
    argmax_pred = pred.idxmax(axis=1).astype(pd.CategoricalDtype(categories=pred.columns))
    argmax_pred = pd.get_dummies(argmax_pred)
    argmax_pred.columns = pred.columns
    argmax_pred.reset_index().to_feather(f'{dirs["gates"]}/{fs}_{panel}.ftr')


def merge_preds(path, fs, pedigree):
    dfs = []
    for i, (parent, kids) in enumerate(pedigree):
        panel = '|'.join(kids)
        df = pd.read_feather(fp := f'{path}/{fs}_{panel}.ftr').set_index('index')
        dfs.append(df.rename(columns={'rest': f'rest_{panel}'}))
        os.remove(fp)
    return pd.concat(dfs, axis=1).fillna(0).reset_index()


if __name__ == '__main__':
    from fastai.basics import glob, torch

    set_seed()

    probe = 'B'
    pedi_name = 'total'  # small total
    pedi = PEDIGREES[pedi_name]

    lmd_11k_dir = '/mnt/data-ssd/cyto/lmd_11k'
    pred_11k_dir = '/mnt/data-ssd/cyto/preds_11k'
    # pred_11k_dir = '/mnt/data-ssd/cyto/test_preds'
    stats_dir = '/mnt/data-ssd/cyto/fastcyto_dirs/extracted_stuff_big'
    # lmd_11k_dir = '/mnt/data-ssd/cyto/fastcyto_dirs/interrater/lmd'
    # pred_11k_dir = '/mnt/data-ssd/cyto/fastcyto_dirs/interrater/gatenet_labels'

    means = pd.read_csv(f'{stats_dir}/means_{probe}.csv', index_col=0)
    stds = pd.read_csv(f'{stats_dir}/stds_{probe}.csv', index_col=0)

    fps = sorted(glob.glob(f'{lmd_11k_dir}/*-{probe}0*'))  # *-{probe}0*
    # fps_all = sorted(glob.glob(f'{lmd_11k_dir}/*-{probe}0*'))  # *-{probe}0*
    # fps = list(set(fps_all) - set(fps))
    skip_fns = {'L': ['07063-00-L001-2'], 'B': ['07040-89-B096-2', '07063-00-B001-2']}[probe]
    fps = [f for i, f in enumerate(fps) if not any(map(f.__contains__, skip_fns))][7000:]
    if probe == 'L':
        fps += [f'{lmd_11k_dir}/06633-00-78L0-1.LMD']

    dirs = {'fcs': lmd_11k_dir, 'labels': None,
            # 'model': f'/home/lfisch/Projects/gatenet/data/models/ms/pedigree_{pedi_name}/{probe}',
            'model': f'/home/lfisch/Projects/gatenet/data/models/ms/pedigree_{pedi_name}/{probe}',
            # 'gates': f'{pred_11k_dir}/pedigree_{pedi_name}/{probe}/argmax_preds',
            # 'preds': f'{pred_11k_dir}/pedigree_{pedi_name}/{probe}/preds',
            'gates': f'{pred_11k_dir}/{probe}/argmax_preds',
            'preds': f'{pred_11k_dir}/{probe}/preds'
            }

    for fp in tqdm(fps):
        predict_file(fp, dirs, pedi, means, stds)
        torch.cuda.empty_cache()
