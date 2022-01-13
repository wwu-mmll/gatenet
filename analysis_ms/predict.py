from fastai.basics import os, np, pd, Path
from tqdm.contrib.concurrent import thread_map

from src.predict import predict
from src.utils import HPARAMS
from analysis_ms.utils import FCS_COL_NAMES, PEDIGREES, NO_REST_PANELS, get_ms_files_df, get_gate_panel


def predict_ms(dirs: dict, idxs: list, probe: str, pedigree: tuple, fcs_col_names: list = FCS_COL_NAMES,
               hparams: dict = HPARAMS, no_rest_panel: list = NO_REST_PANELS):
    gates_dir = dirs['gates']
    parent_id_groups = None
    for i, (parent, kids) in enumerate(pedigree):
        gate_panel = '_'.join(kids)
        dirs.update({'gates': None if parent is None else gates_dir})
        df = get_ms_files_df(dirs, idxs, probe, gate_panel, get_gate_panel(parent, pedigree))
        dirs.update({'gates': gates_dir})
        print(f'PREDICT {gate_panel}')

        vocab = kids if gate_panel in no_rest_panel else kids + ['rest']
        ds_kwargs = {'pre_gate': parent, 'vocab': vocab, 'fcs_col_names': fcs_col_names}

        model_path = f'{dirs["model"]}/{gate_panel}.pth'
        preds = predict(model_path, df, hparams, ds_kwargs=ds_kwargs, outputs=('preds',))
        if parent is None:
            parent_id_groups = preds.groupby('id').groups
        save_preds(preds, dirs, parent_id_groups, panel=gate_panel)
    merge_preds(dirs, pedigree, df)


def save_preds(pred: pd.DataFrame, dirs: dict, parent_id_groups: dict, panel: str):
    id_groups = pred.groupby('id').groups
    pred.drop(columns=['id'], inplace=True)
    pred.columns = [c.replace('_pred', '') for c in pred.columns]
    for mod in ['preds', 'gates']:
        Path(dirs[mod]).mkdir(exist_ok=True, parents=True)
        if mod == 'gates' and len(pred) != 0:
            pred = pred.idxmax(axis=1).astype(pd.CategoricalDtype(categories=pred.columns))
            pred = pd.get_dummies(pred)
        # params = [{'id': id, 'parent_idxs': parent_idxs, 'pred': pred, 'parent_id_groups': parent_id_groups, 'id_groups': id_groups, 'save_dir': dirs[mod], 'panel': panel} for id, parent_idxs in parent_id_groups.items()]
        # res = thread_map(save_splitted_idxs_parallel, params)
        for id, parent_idxs in parent_id_groups.items():
            save_splitted_idxs(id, parent_idxs, pred, id_groups, dirs[mod], panel)


# def save_splitted_idxs_parallel(kwargs):
#     save_splitted_idxs(**kwargs)


def save_splitted_idxs(id, parent_idxs, pred, id_groups, save_dir, panel):
    if id in [k for k, _ in id_groups.items()]:
        id_pred = pred.loc[id_groups[id]]
        id_pred.index = id_pred.index - parent_idxs[0]
        if len(id_groups[id]) != len(parent_idxs):
            id_pred = id_pred.reindex(np.arange(len(parent_idxs)), fill_value=0)
    else:
        id_pred = pd.DataFrame(0, index=np.arange(len(parent_idxs)), columns=pred.columns.tolist())
    id_pred.to_csv(f'{save_dir}/{id}_{panel}.csv', index=False)


def merge_preds(dirs: dict, pedigree: tuple, df):
    filestems = [Path(f).stem for f in df['fcs']]
    for mods in ['preds', 'gates']:
        for f in filestems:
            dfs = []
            for parent, kids in pedigree:
                panel = '_'.join(kids)
                fp = f'{dirs[mods]}/{f}_{panel}.csv'
                dfs.append(pd.read_csv(fp).rename(columns={'rest': f'rest_{panel}'}))
                os.remove(fp)
            merged_df = pd.concat(dfs, axis=1)
            merged_df.to_csv(f'{dirs[mods]}/{f}.csv', index=False)


if __name__ == '__main__':
    idxs = [i for i in range(13)]  # [i for i in range(8250, 8300) if i != 8256]
    probe = 'L'
    pedigree_name = 'total'  # small total
    pedigree = PEDIGREES[pedigree_name]

    lmd_11k_dir = '/mnt/data-ssd/cyto/lmd_11k'
    pred_11k_dir = '/mnt/data-ssd/cyto/predicted_labels_11k'

    dirs = {'fcs': lmd_11k_dir,
            # 'fcs': '/mnt/data-ssd/cyto/fastcyto_dirs/extracted_stuff_big/lmds',
            'labels': None,
            'model': f'/home/lfisch/Projects/gatenet/data/models/ms/pedigree_{pedigree_name}/{probe}',  # test pedigree_{pedigree_name}
            'gates': f'{pred_11k_dir}/pedigree_{pedigree_name}/{probe}/argmax_preds',
            'preds': f'{pred_11k_dir}/pedigree_{pedigree_name}/{probe}/preds'}

    predict_ms(dirs, idxs, probe, pedigree)
