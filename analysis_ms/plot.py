import mpl_scatter_density
from fastai.basics import plt
from analysis_ms.utils import PEDIGREES, MARKER_NRS, CELL_NAMES


def plot_gating_strategy(df_b, df_l):
    f, axs = plt.subplots(4, 5, figsize=(18, 14))
    gating_sample(df_b, PEDIGREES['small'], axs=axs[:2])
    gating_sample(df_l, PEDIGREES['small'], axs=axs[2:])
    axs[0, 0].set_title('Patient blood (PB)', fontweight='bold', fontsize=15, x=0.2, y=1.15)
    axs[2, 0].set_title('Cerebrospinal fluid (CSF)', fontweight='bold', fontsize=15, x=0.37, y=1.15)
    f.tight_layout()
    return f


def gating_sample(df, pedigree, marker_nrs=MARKER_NRS, cell_names=CELL_NAMES, axs=None):
    if axs is None:
        n_subplots = (2, 5) if len(pedigree) == 10 else (3, 6)
        fsize = (16, 8) if len(pedigree) == 10 else (24, 13)
        f, axs = plt.subplots(*n_subplots, figsize=fsize)  # , subplot_kw={'projection': 'scatter_density'})
    for (p, k), ax in zip(pedigree, axs.flatten()):
        gating_window(df, p, k, marker_nrs, cell_names, ax)
    return axs


def gating_window(df, parent, kids, marker_nrs=MARKER_NRS, cell_names=CELL_NAMES, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    marker_nr_x, marker_nr_y = tuple(marker_nrs[tuple(kids)])
    marker_x, marker_y = df.columns[marker_nr_x], df.columns[marker_nr_y]
    df = df if parent is None else df[df[parent] > .5]
    colors = ['gray', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for i, k in enumerate(['Rest'] + kids):
        df_k = df[df[kids].le(.5).all(1)] if k == 'Rest' else df[df[k] > .5]
        # df_k = df_k.sample(2000) if len(df_k) > 2000 else df_k
        plot_name = cell_names[k] if k in cell_names else k
        # ax.scatter_density(df_k[marker_x], df_k[marker_y], label=plot_name, color=colors[i], alpha=1., dpi=50)
        ax.scatter(df_k[marker_x], df_k[marker_y], label=plot_name, s=4., color=colors[i], alpha=.3)
    xlim = (0, df[marker_x].max()) if 'Beads' in kids else (0, 1023)
    ylim = (0, 1023)  # (df[marker_y].min(), df[marker_y].max())
    ax = gating_window_axis(ax, marker_x, marker_y, xlim, ylim, parent, cell_names)
    return ax


def gating_window_old(df, parent, kids, marker_nrs=MARKER_NRS, cell_names=CELL_NAMES, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    marker_nr_x, marker_nr_y = tuple(marker_nrs[tuple(kids)])
    marker_x, marker_y = df.columns[marker_nr_x], df.columns[marker_nr_y]
    df = df if parent is None else df[df[parent] > .5]
    for k in ['Rest'] + kids:
        df_k = df[df[kids].le(.5).all(1)] if k == 'Rest' else df[df[k] > .5]
        # df_k = df_k.sample(2000) if len(df_k) > 2000 else df_k
        markercolor = 'gray' if k == 'Rest' else None
        plot_name = cell_names[k] if k in cell_names else k
        ax.scatter(df_k[marker_x], df_k[marker_y], label=plot_name, s=4, color=markercolor, alpha=.1)
    xlim = (0, df[marker_x].max()) if 'Beads' in kids else (0, 1023)
    ylim = (0, 1023)  # (df[marker_y].min(), df[marker_y].max())
    ax = gating_window_axis(ax, marker_x, marker_y, xlim, ylim, parent, cell_names)
    return ax


def gating_window_axis(ax, marker_x, marker_y, xlim, ylim, parent, cell_names):
    ax.legend()
    ax.set_xlim([*xlim])
    ax.set_ylim([*ylim])
    ax.set_xlabel(marker_x)
    ax.set_ylabel(marker_y)
    if parent is not None:
        plot_name = cell_names[parent] if parent in cell_names else parent
        ax.set_title(plot_name, fontweight='bold')
    return ax


if __name__ == '__main__':
    import os
    from glob import glob
    from tqdm import tqdm
    from pathlib import Path

    from src.data.utils import fcs_label_df

    from analysis_ms.utils import FCS_COL_NAMES
    PLOT_DIR = '/home/lfisch/Projects/gatenet/data/plots/'
    GATING_PATH = '/mnt/data-ssd/cyto/fastcyto_dirs/extracted_stuff_big'
    pedigree_name = 'small'  # small total
    pedi = PEDIGREES[pedigree_name]
    
    # import pandas as pd
    # df = pd.read_feather('/mnt/data-ssd/cyto/test_preds/preds/00012-00-L001-1.ftr')
    # print(df[df['CD45+'] > .5][['Granulos', 'Lymphos', 'Monos', 'rest_Granulos|Monos|Lymphos']].iloc[:40])
    # err

    # PB/CSF GATING
    # patient = 'BK23081973'
    # lmd_dir = f'{gating_path}/lmds'
    # label_dir = f'{gating_path}/labels'
    # df_pb = fcs_label_df(f'{patient}B00', lmd_dir, label_dir)
    # df_csf = fcs_label_df(f'{patient}L00', lmd_dir, label_dir)
    # fig = plot_gating_strategy(df_pb, df_csf)
    # plt.savefig(f'{PLOT_DIR}/gating_strategy.png')  # .pdf

    # PRED GATING
    probe = 'B'
    # lmd_dir = '/mnt/data-ssd/cyto/lmd_11k'
    # label_dir = f'/mnt/data-ssd/cyto/preds_11k/{probe}/argmax_preds'
    # patients = [Path(f).stem for f in sorted(glob(f'{label_dir}/*-{probe}0*'))][:1]  # [15:17]
    lmd_dir = '/mnt/data-ssd/cyto/fastcyto_dirs/interrater/lmd'
    label_dir = f'/mnt/data-ssd/cyto/fastcyto_dirs/interrater/feather/intersect'
    patients = [Path(f).stem for f in sorted(glob(f'{label_dir}/*{probe}00*'))]
    # patients = ['03115-01-B002-1']
    subdir = f'interrater/intersect'
    print(len(patients))

    for patient in tqdm(patients):
        df_pred = fcs_label_df(patient, lmd_dir, label_dir)
        gating_sample(df_pred, pedigree=pedi)
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'{PLOT_DIR}/{subdir}/{patient}.png')

    # ANDI GATING
    # import fcsparser
    # import pandas as pd
    #
    # lmd_dir = f'{GATING_PATH}/lmds'
    # label_dir = f'{GATING_PATH}/labels'
    # lmd_fs, label_fs = sorted(glob(f'{lmd_dir}/*')), sorted(glob(f'{label_dir}/*'))
    # df = pd.DataFrame(0, [Path(f).stem for f in lmd_fs], ["_".join(kids) for _, kids in pedi]).to_csv('qc.csv')
    # for lmd_f, label_f in tqdm(zip(lmd_fs, label_fs)):
    #     fcs = fcsparser.parse(lmd_f, reformat_meta=True)[1]
    #     fcs.columns = FCS_COL_NAMES
    #     label = pd.read_csv(label_f)
    #     df_pred = pd.concat([fcs, label], axis=1)
    #     gating_sample(df_pred, pedigree=pedi)
    #     plt.tight_layout()
    #     plt.savefig(f'{PLOT_DIR}/andi/{pedigree_name}/{Path(label_f).stem}.png')
    #     plt.clf()
