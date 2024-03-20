from collections import ChainMap
from tqdm.contrib.concurrent import thread_map
from fastai.basics import pd, glob, Path

from utils import PEDIGREES


def counts_to_per_ml(counts: pd.DataFrame):
    counts = counts.copy(deep=True)
    beads = counts.pop('Beads')
    vol = counts.pop('harvest_volume')
    counts = counts.div(beads, axis=0).div(vol, axis=0)
    counts.loc[counts.index.str.contains('L0')] *= 5e4
    counts.loc[counts.index.str.contains('B0')] *= 2e5
    return counts


def counts_to_relative(counts: pd.DataFrame, pedigree: tuple, in_percent: bool = True):
    counts = counts.copy(deep=True)
    counts.drop(columns=['Beads', 'harvest_volume'], inplace=True)
    for parent, kids in pedigree[::-1]:
        if parent is not None:
            counts[kids] = counts[kids].div(counts[parent], axis=0)
    counts.drop(columns=['CD45+'], inplace=True)
    counts = 100. * counts if in_percent else counts
    return counts


def merge_table_11k(counts: pd.DataFrame, table_11k: pd.DataFrame, keep_cols: list = ['harvest_volume'], file_col: str = 'New_File_Name'):
    table_11k['File_Stem'] = table_11k[file_col].str.slice(0, -4)
    table_11k = table_11k[['File_Stem'] + keep_cols]
    counts['File_Stem'] = counts.index
    counts = pd.merge(table_11k, counts, how='left', on='File_Stem')
    counts = counts.set_index('File_Stem').sort_index()
    return counts


def directory_label_counts(paths: list, pedigree: tuple):
    filepaths = []
    for p in paths:
        filepaths += glob.glob(f'{p}/*.ftr')
    filepaths = sorted(filepaths)
    # filepaths = filepaths[4090:4100]

    args_list = [(f, pedigree) for f in filepaths]
    def label_count_file_kwargs(args): return label_counts_file(*args)
    counts = thread_map(label_count_file_kwargs, args_list)
    counts = dict(ChainMap(*counts))
    return pd.DataFrame(counts).T.sort_index()


def label_counts_file(filepath: str, pedigree: tuple):
    df = pd.read_feather(filepath)
    return {Path(filepath).stem: label_counts(df, pedigree)}


def label_counts(df: pd.DataFrame, pedigree: tuple):
    counts = {}
    for parent, kids in pedigree:
        for k in kids:
            counts.update({k: df[k].sum()})
    return counts


if __name__ == '__main__':
    run = 'saskia'
    probe = 'L'
    pedi_name = 'total'  # small total
    pedi = PEDIGREES[pedi_name]
    counts_res_dir = f'/home/lfisch/Projects/gatenet/data/counts/ms/{run}'

    table_11k = pd.read_csv('/home/lfisch/Projects/gatenet/analysis_ms/archive/final_pseudo_new.csv')

    pred_dirs = [f'/mnt/data-ssd/cyto/preds_11k/L/argmax_preds',
                 f'/mnt/data-ssd/cyto/preds_11k/B/argmax_preds']  # preds

    counts = directory_label_counts(pred_dirs, pedi)
    counts = merge_table_11k(counts, table_11k)

    valid_samples = pd.read_csv('/mnt/data-ssd/cyto/lmd_11k_columns_filtered.csv').id
    only_valid_samples = True
    # err
    
    absolute_counts = counts.drop(columns=['harvest_volume']).astype('Int64')
    if only_valid_samples:
        absolute_counts = absolute_counts[counts.index.isin(valid_samples)]
    absolute_counts.to_csv(f'{counts_res_dir}/absolute.csv')

    per_ml = counts_to_per_ml(counts)
    if only_valid_samples:
        per_ml = per_ml[counts.index.isin(valid_samples)]
    per_ml.to_csv(f'{counts_res_dir}/per_ml.csv', float_format='%.2f')

    relative_counts = counts_to_relative(counts, pedi)
    if only_valid_samples:
        relative_counts = relative_counts[counts.index.isin(valid_samples)]
    relative_counts.to_csv(f'{counts_res_dir}/relative.csv', float_format='%.2f')
