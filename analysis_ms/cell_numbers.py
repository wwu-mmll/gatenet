from tqdm import tqdm
from collections import ChainMap
from tqdm.contrib.concurrent import thread_map
from fastai.basics import os, np, pd, glob, Path

from utils import PEDIGREES


def counts_to_per_ml(counts: pd.DataFrame):
    beads = counts.pop('Beads')
    vol = counts.pop('harvest_volume')
    counts = counts.div(beads, axis=0).div(vol, axis=0)
    counts.loc[counts.index.str.contains('L0')] *= 5e4
    counts.loc[counts.index.str.contains('B0')] *= 2e5
    return counts


def merge_table_11k(counts: pd.DataFrame, table_11k: pd.DataFrame, keep_cols: list = ['harvest_volume'], file_col: str = 'New_File_Name'):
    table_11k['File_Stem'] = table_11k[file_col].str.slice(0, -4)
    table_11k = table_11k[['File_Stem'] + keep_cols]
    counts['File_Stem'] = counts.index
    counts = pd.merge(table_11k, counts, how='left', on='File_Stem')
    return counts.set_index('File_Stem').sort_index()


def directory_label_counts(paths: list, pedigree: tuple, percent: bool):
    filepaths = []
    for p in paths:
        filepaths += glob.glob(f'{p}/*1.csv')
    filepaths = sorted(filepaths)[:1000]

    args_list = [(f, pedigree, percent) for f in filepaths]
    def label_count_file_kwargs(args): return label_counts_file(*args)
    counts = thread_map(label_count_file_kwargs, args_list)
    counts = dict(ChainMap(*counts))
    return pd.DataFrame(counts).T.sort_index()


def label_counts_file(filepath: str, pedigree: tuple, percent: bool):
    df = pd.read_csv(filepath)
    return {Path(filepath).stem: label_counts(df, pedigree, percent)}


def label_counts(df: pd.DataFrame, pedigree: tuple, percent: bool):
    counts = {}
    for parent, kids in pedigree:
        if not (percent and parent is None):
            parent_count = .01 * df[parent].sum() if percent else 1
            for k in kids:
                count = df[k].sum() / parent_count if percent else df[k].sum()
                counts.update({k: count})
    return counts


if __name__ == '__main__':
    probe = 'L'
    pedi_name = 'total'  # small total
    pedi = PEDIGREES[pedi_name]
    counts_res_dir = '/home/lfisch/Projects/gatenet/data/counts/ms'

    table_11k = pd.read_csv('/home/lfisch/Projects/gatenet/analysis_ms/archive/final_pseudo_new.csv')

    pred_dirs = [f'/mnt/data-ssd/cyto/predicted_labels_11k/pedigree_{pedi_name}/L/argmax_preds',
                 f'/mnt/data-ssd/cyto/predicted_labels_11k/pedigree_{pedi_name}/B/argmax_preds']  # preds

    absolute_counts = directory_label_counts(pred_dirs, pedi, False)
    absolute_counts = merge_table_11k(absolute_counts, table_11k)
    absolute_counts.drop(columns=['harvest_volume']).to_csv(f'{counts_res_dir}/absolute.csv')

    per_ml = counts_to_per_ml(absolute_counts)
    per_ml.to_csv(f'{counts_res_dir}/per_ml.csv')

    relative_counts = directory_label_counts(pred_dirs, pedi, True)
    relative_counts = merge_table_11k(relative_counts, table_11k)
    relative_counts.drop(columns=['harvest_volume']).to_csv(f'{counts_res_dir}/relative.csv')
