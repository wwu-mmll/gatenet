from fastai.basics import plt
from utils import PEDIGREE, MARKERS, CELL_NAMES
PANEL_AX_LIMITS = {('CD45+'): ((0., .0), (.0, .0)),
                   ('Granulos', 'Monos', 'Lymphos'): ((.0, .0), (.0, .0))}


def plot_gating_strategy(df_b, df_l):
    f, axs = plt.subplots(4, 5, figsize=(18, 14))
    gating_sample(df_b, axs=axs[:2])
    gating_sample(df_l, axs=axs[2:])
    axs[0, 0].set_title('Patient blood (PB)', fontweight='bold', fontsize=15, x=0.2, y=1.15)
    axs[2, 0].set_title('Cerebrospinal fluid (CSF)', fontweight='bold', fontsize=15, x=0.37, y=1.15)
    #correct_ax_limits(axs)
    f.tight_layout()
    return f


def gating_sample(df, pedigree=PEDIGREE, markers=MARKERS, cell_names=CELL_NAMES, axs=None):
    if axs is None:
        f, axs = plt.subplots(2, 5, figsize=(16, 8))
    for (p, k), ax in zip(pedigree, axs.flatten()):
        gating_window(df, p, k, markers, cell_names, ax)
    return axs


def gating_window(df, parent, kids, markers=MARKERS, cell_names=CELL_NAMES, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    marker_x, marker_y = tuple(markers[tuple(kids)])
    df = df if parent is None else df[df[parent] == 1]
    for k in kids + ['Rest']:
        df_k = df[df[kids].eq(0).all(1)] if k == 'Rest' else df[df[k] == 1]
        markercolor = 'gray' if k == 'Rest' else None
        ax.scatter(df_k[marker_x], df_k[marker_y], label=cell_names[k], s=4, color=markercolor, alpha=.3)
    # xlim = (df[marker_x].min(), df[marker_x].max())
    # ylim = (df[marker_y].min(), df[marker_y].max())
    xlim = (0, df[marker_x].max()) if 'Beads' in kids else (0, 1023)
    ylim = (0, 1023)
    ax = gating_window_axis(ax, marker_x, marker_y, xlim, ylim, parent, cell_names)
    return ax


def gating_window_axis(ax, marker_x, marker_y, xlim, ylim, parent, cell_names):
    ax.legend()
    ax.set_xlim([*xlim])
    ax.set_ylim([*ylim])
    ax.set_xlabel(marker_x)
    ax.set_ylabel(marker_y)
    if parent is not None:
        ax.set_title(cell_names[parent], fontweight='bold')
    return ax


if __name__ == '__main__':
    from src.data.utils import fcs_label_df
    PLOT_DIR = '/home/lfisch/Projects/gatenet/data/plots'

    # PB/CSF GATING
    patient = 'BK23081973'
    gating_path = '/mnt/data-ssd/cyto/fastcyto_dirs/extracted_stuff_big'
    lmd_dir = f'{gating_path}/lmds'
    label_dir = f'{gating_path}/labels'
    df_pb = fcs_label_df(f'{patient}B00', lmd_dir, label_dir)
    df_csf = fcs_label_df(f'{patient}L00', lmd_dir, label_dir)
    # fig = plot_gating_strategy(df_pb, df_csf)
    # plt.savefig(f'{PLOT_DIR}/gating_strategy.png')  # .pdf

    # PRED GATING
    patients = ['00000-00-L001-1', '00001-00-L001-1', '00002-00-L001-1', '00003-00-L001-1', '00004-00-L001-1',
                '00005-00-L001-1', '00007-00-L001-1', '00008-00-L001-1', '00009-00-L001-1', '00010-00-L001-1']
    lmd_dir = '/mnt/data-ssd/cyto/lmd_11k'
    # label_dir = '/spm-data/Scratch/spielwiese_lukas/data/cyto/predicted_labels_11k/pedigree_total/L/argmax_preds'
    label_dir = '/mnt/data-ssd/cyto/predicted_labels_11k/pedigree_total_test/L/argmax_preds'
    # for patient in patients:
    #     df_pred = fcs_label_df(patient, lmd_dir, label_dir)
    #     gating_sample(df_pred, pedigree=PEDIGREE)
    #     plt.tight_layout()
    #     plt.savefig(f'{PLOT_DIR}/pred/{patient}.png')

    # ANDI GATING
    from glob import glob
    from pathlib import Path
    lmd_dir = f'{gating_path}/lmds'
    label_dir = f'{gating_path}/labels'
    patients = [Path(f).stem for f in glob(f'{lmd_dir}/*') if len(Path(f).stem) == 14 and 'L0' in f]
    for patient in patients:
        df_pred = fcs_label_df(patient, lmd_dir, label_dir)
        try:
            gating_sample(df_pred, pedigree=PEDIGREE)
            plt.tight_layout()
            plt.savefig(f'{PLOT_DIR}/andi/{patient}.png')
        except:
            pass
