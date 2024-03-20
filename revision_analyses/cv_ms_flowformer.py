import glob
import torch
import fcsparser
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torcheval.metrics.functional import multiclass_f1_score
from fastai.basics import accuracy, BalancedAccuracy, F1Score
from fastai.data.all import warnings, Categorize, CategoryMap

from src.data.files import get_files_df
from src.eval import evaluate_cv
from paper_analyses.utils import FCS_COL_NAMES, PEDIGREE
from revision_analyses.flowformer import SetTransformer


def cv(df, ds_kwargs, folds):
    cv_result = []
    for i, (idxs_t, idxs_v) in enumerate(folds):
        df_train, df_valid = df.iloc[idxs_t], df.iloc[idxs_v]
        fold_result = train_transformer(df_train, df_valid, ds_kwargs)
        # fold_result.to_csv('/home/lfisch/Projects/gatenet_old/data/results/ms/junk.csv')
        fold_result['fold'] = i
        cv_result.append(fold_result)
    return pd.concat(cv_result)


def train_transformer(df_train, df_valid, ds_kwargs):
    ds_train = FCSDataset(df_train.fcs, df_train.labels, df_train.gates if 'gates' in df_train else None, **ds_kwargs)
    ds_valid = FCSDataset(df_valid.fcs, df_valid.labels, df_valid.valid_gates if 'gates' in df_train else None, **ds_kwargs)
    dl_train = DataLoader(ds_train, batch_size=1)
    dl_valid = DataLoader(ds_valid, batch_size=1, shuffle=False)
    model = SetTransformer(num_inds=16, dim_hidden=32, num_heads=4, layer_norm=True, hidden_layers=3,
                           residual=False, _num_markers=ds_train.fcs_dfs[0].shape[1], _sequence_length=False,
                           mode='autoencoder', dim_output=len(ds_kwargs['vocab']))  # mode='binary'
    model = model.cuda()
    softmax = torch.nn.Softmax(dim=2)
    optim = torch.optim.Adam(model.parameters(), lr=.001, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=10, eta_min=.0002)
    loss_func = torch.nn.CrossEntropyLoss()  # torch.nn.BCELoss()
    progress_bar = tqdm(range(2))  # tqdm(range(2))
    for epoch in progress_bar:
        for fcs, label in dl_train:
            fcs, label = fcs.cuda(), label.cuda()
            optim.zero_grad()
            label_pred = softmax(model(fcs))
            loss = loss_func(label_pred[0], label[0])
            #tqdm.write(f'Loss: {loss.cpu().item()}')
            loss.backward()
            optim.step()
            scheduler.step()
        progress_bar.set_description(f'Loss: {loss.cpu().item()}')

    results = {}
    with torch.no_grad():
        for fps, (fcs, label) in zip(df_valid.fcs, dl_valid):
            fcs, label = fcs.cuda(), label.cuda()
            label_pred = softmax(model(fcs))
            # print(fps)
            # print(label_pred[0].sum(0).cpu().numpy())
            # print(torch.unique(label[0], return_counts=True))
            # print(torch.unique(label_pred[0].argmax(1), return_counts=True))
            results.update({Path(fps).stem: get_metrics(label_pred[0], label[0], len(ds_kwargs['vocab']))})
    return pd.DataFrame(results).T


def get_metrics(preds, targs, n_classes):
    BACC = BalancedAccuracy()
    acc = accuracy(preds, targs).item() if n_classes == 2 else 0.
    bacc = BACC(preds.argmax(1), targs) if n_classes == 2 else 0.
    return {'F1 score (weighted)': multiclass_f1_score(preds, targs, num_classes=n_classes, average='weighted').item(),
            'F1 score (unweighted)': multiclass_f1_score(preds, targs, num_classes=n_classes, average='macro').item(),
            'Accuracy': acc, 'Balanced Accuracy': bacc}


def do_pedigree(df, pedigree, dirs, hparams, means, stds, folds):
    result_dfs = []
    for parent, kids in pedigree:
        panel = '|'.join(kids)
        print(f'CV {panel}')
        eval_dir = None if dirs['eval'] is None else dirs['eval'] + f'/{panel}'
        result = do_cv(df, parent, kids, eval_dir, hparams, means, stds, folds)

        # if dirs['pred'] is not None:
        #     save_preds(result, panel, dirs['pred'])
        result.insert(0, 'panel', panel)
        print(result.iloc[:, :-1])
        result_dfs.append(result.set_index('panel', append=True))
    return pd.concat(result_dfs)

def do_cv(df, parent, kids, eval_dir, hparams, means, stds, folds):
    panel = '|'.join(kids)
    ds_kwargs = {'pre_gate': parent, 'vocab': kids + ['rest'],
                 'fcs_col_names': FCS_COL_NAMES, 'fcs_cols': FCS_COL_NAMES[:-1],
                 'fcs_means': means.loc[panel], 'fcs_stds': stds.loc[panel]}
    result = cv(df, ds_kwargs, folds)
    # if eval_dir is not None:
    #     Path(eval_dir).mkdir(exist_ok=True, parents=True)
    #     evaluate_cv(result, eval_dir)
    return result


def get_df(dirs, probe='L'):
    dfs = []
    for label_dir in dirs['label']:
        rater_df = get_files_df(dirs['fcs'], label_dir, label_dir, substr=f'{probe}00')
        rater_df['valid_gates'] = dirs['gate'] + '/' + rater_df.gates.str.rsplit('/', 1, expand=True)[1]
        dfs.append(rater_df)
    return pd.concat(dfs).sort_index().reset_index(drop=True)


def save_preds(result, panel, pred_dir):
    Path(pred_dir).mkdir(exist_ok=True)
    for id in result.id.unique():
        id_results = result[result.id == id]
        id_results.reset_index().to_feather(f'{pred_dir}/{Path(id).stem}_{panel}.ftr')


def get_multifolds(folds, experts):
    multifolds = []
    for i, (train_idxs, valid_idxs) in enumerate(folds):
        t_idxs = np.concatenate([np.array([]) if e is None else (train_idxs * len(experts)) + i for i, e in enumerate(experts)])
        v_idxs = valid_idxs * len(experts)
        multifolds.append((t_idxs, v_idxs))
    return multifolds


class FCSDataset(Dataset):
    def __init__(self, fcs_filepaths, label_filepaths, gate_filepaths=None, pre_gate=None, vocab=None,
                 fcs_cols=None, fcs_col_names=None, multiclass_col=None, fcs_means=None, fcs_stds=None):
        self.vocab = Categorize(vocab=vocab).vocab
        self.fcs_dfs = [self.load_table(f, fcs_cols, fcs_col_names) for f in fcs_filepaths]
        if fcs_means is not None and fcs_stds is not None:
            self.fcs_dfs = [self.normalize(df, fcs_means, fcs_stds) for df in self.fcs_dfs]
        self.label_dfs = [self.load_table(f) for f in label_filepaths]
        if pre_gate is not None:
            gate_dfs = self.label_dfs if gate_filepaths is None else [self.load_table(f) for f in gate_filepaths]
            gate_idxs = [df[pre_gate] == 1 for df in gate_dfs]
            self.fcs_dfs = [df.loc[idxs] for df, idxs in zip(self.fcs_dfs, gate_idxs)]
            self.label_dfs = [df.loc[idxs] for df, idxs in zip(self.label_dfs, gate_idxs)]
        self.label_dfs = self.prepare_labels(self.label_dfs, multiclass_col)

    def __len__(self):
        return len(self.fcs_dfs)

    def __getitem__(self, idx):
        return self.fcs_dfs[idx].values.astype(np.float32), self.label_dfs[idx].values

    def load_table(self, filepath, cols=None, col_names=None):
        df = self.read_file(filepath)
        df.columns = df.columns if col_names is None else col_names
        return df if cols is None else df[cols]

    def prepare_labels(self, dfs, multiclass_col):
        out_dfs = []
        for df in dfs:
            if multiclass_col is None:
                df = df[[v for v in self.vocab if v != 'rest']]
                df = self.add_rest_class(df.copy()) if 'rest' in self.vocab else df
                df = self.reverse_one_hot_encode(df, self.vocab)
            else:
                df = df[multiclass_col] if isinstance(multiclass_col, str) else df.iloc[:, multiclass_col]
            out_dfs.append(df)
        return out_dfs

    def add_rest_class(self, df):
        rest = df.eq(0).all(1).astype(int)
        if rest.sum() > 0:
            df['rest'] = rest
        else:
            warnings.warn('No zero labels. Rest class could not be added. Will be removed in vocab', UserWarning)
            self.vocab = Categorize(vocab=[v for v in self.vocab if v != 'rest']).vocab
        return df

    @staticmethod
    def read_file(filepath):
        suffix = Path(filepath).suffix
        if suffix in ('.fcs', '.lmd', '.LMD'):
            df = fcsparser.parse(filepath, reformat_meta=True)[1]
        elif suffix == '.csv':
            df = pd.read_csv(filepath)
        elif suffix == '.ftr':
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

    @staticmethod
    def normalize(df, means, stds):
        return (df - means) / stds


if __name__ == '__main__':
    # rename_dict = {'CD45+': 'Leuko', 'Granulos|Monos|Lymphos': 'Granulo-Lympho-Mono', 'NKT|NK|T': 'NK-NKT-T',
    #                'B|Plasma cells': 'B-Plasma', 'CD56b|CD56dCD16+': 'CD56CD16',
    #                'CD14+CD16+|CD14+CD16-|CD14low CD16high': 'CD14CD16', 'CD4+|CD8+': 'CD4CD8',
    #                'CD4+HLADR+': 'CD4+HLADR+', 'CD8+HLADR+': 'CD8+HLADR+'}
    #
    # df = pd.read_csv('/home/lfisch/Projects/gatenet_old/data/results/ms/flowformer_results_backup.csv', index_col=0)
    # df['panel'] = df.panel.map(rename_dict)
    # df = df.sort_index()
    # df = df.iloc[:, :-3]
    # df.to_csv('/home/lfisch/Projects/gatenet_old/data/results/ms/flowformer_results.csv')

    path = '/mnt/data-ssd/cyto/fastcyto_dirs/interrater'
    # lmd_fps = sorted(glob.glob(f'{path}/lmd/*'))[:5]
    # label_fps = sorted(glob.glob(f'{path}/feather/andi/*.ftr'))[:5]
    #
    # ds = FCSDataset(lmd_fps, label_fps, pre_gate='Lymphos', vocab=['NKT', 'NK', 'T', 'rest'],
    #                 fcs_col_names=FCS_COL_NAMES, fcs_cols=FCS_COL_NAMES[:-1])
    # dl = DataLoader(ds, batch_size=1, shuffle=True)
    #
    # for x, y in dl:
    #     print(x.shape, y.shape)
    #     print(x.dtype, y.dtype)


    # pedi = ((None, ['CD45+']), ('CD45+', ['Granulos', 'Monos', 'Lymphos']))  # PEDIGREE
    # pedi = (('Lymphos', ['B', 'Plasma cells']), ('NK', ['CD56b', 'CD56dCD16+']))#, ('NK', ['CD56b', 'CD56dCD16+']), ('CD45+', ['Granulos', 'Monos', 'Lymphos']))
    pedi = PEDIGREE
    path = '/mnt/data-ssd/cyto/fastcyto_dirs/interrater'
    experts = ['andi', 'christine', 'leonie', 'saskia']
    # experts = ['andi']
    label_dirs = [f'{path}/feather/{expert}' for expert in experts]
    stats_dir = '/mnt/data-ssd/cyto/fastcyto_dirs/extracted_stuff_big'
    dirs = {'fcs': f'{path}/lmd', 'label': label_dirs, 'gate': f'{path}/feather/intersect'}

    hparams = {}
    res_all = []
    for probe in ['L', 'B']:
        dirs.update({'eval': f'/home/lfisch/Projects/gatenet_old/data/results/ms/{probe}/trafo',
                     'pred': f'{path}/cv_preds/{probe}/trafo'})
        df_ = get_df(dirs, probe=probe)

        folds_ = np.load(f'/home/lfisch/Projects/gatenet_old/paper_analyses/folds_{probe}.npy', allow_pickle=True)
        multifolds = get_multifolds(folds_, experts)

        means = pd.read_csv(f'{stats_dir}/means_{probe}.csv', index_col=0)
        stds = pd.read_csv(f'{stats_dir}/stds_{probe}.csv', index_col=0)
        res = do_pedigree(df_, pedi, dirs, hparams, means, stds, multifolds)
        res_all.append(res)
    pd.concat(res_all).to_csv('/home/lfisch/Projects/gatenet_old/data/results/ms/flowformer_results.csv')
