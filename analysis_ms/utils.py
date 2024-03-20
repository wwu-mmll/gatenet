from fastai.basics import np

from src.data.files import get_files_df
PEDIGREE_TOTAL = ((None, ['CD45+']),
                  ('CD45+', ['Granulos', 'Monos', 'Lymphos']),
                  ('Lymphos', ['NKT', 'NK', 'T']),
                  ('Lymphos', ['B', 'Plasma cells']),
                  ('Lymphos', ['B-_T-cell doubletes']),
                  ('NK', ['CD56bCD16d', 'CD56dCD16+', 'CD56bCD16-', 'CD56dCD16-']),
                  (None, ['Beads']),
                  ('Monos', ['CD14+CD16+', 'CD14+CD16-', 'CD14low CD16high']),
                  ('Monos', ['D', 'F']),
                  ('T', ['CD4+', 'CD8+', 'A--', 'CD4+CD8+']),
                  ('CD4+', ['CD4+HLADR+']),
                  ('CD8+', ['CD8+HLADR+']),
                  ('CD4+CD8+', ['CD4+CD8+HLADR+']),
                  ('T', ['T+HLADR+']),
                  ('D', ['CD16-HLADR+']),
                  ('F', ['CD16+HLADR+']))
PEDIGREE_SMALL = PEDIGREE_TOTAL[:4] + (('NK', ['CD56bCD16d', 'CD56dCD16+']),) + PEDIGREE_TOTAL[6:8] + \
                 (('T', ['CD4+', 'CD8+']),) + PEDIGREE_TOTAL[10:12]
PEDIGREES = {'small': PEDIGREE_SMALL, 'total': PEDIGREE_TOTAL}
FCS_COL_NAMES = ['FSC', 'SSC', 'CD14-FITC', 'CD138-PE', 'HLA DR-ECD', 'CD3-PC5.5', 'CD56-PC7',
                 'CD4-APC', 'CD19-A700', 'CD16-A750', 'CD8-PB', 'CD45-KrOrange', 'TIME']
NO_REST_PANELS = ['CD4+|CD8+|A--|CD4+CD8+', 'CD56bCD16d|CD56dCD16+|CD56bCD16-|CD56dCD16-']
MARKER_NRS = {('CD45+', ): (11, 0), ('Beads', ): (12, 4), ('Granulos', 'Monos', 'Lymphos'): (2, 1),
              ('NKT', 'NK', 'T'): (5, 6), ('B', 'Plasma cells'): (3, 8), ('B-_T-cell doubletes', ): (3, 8),
              ('CD56bCD16d', 'CD56dCD16+'): (9, 6), ('CD56bCD16d', 'CD56dCD16+', 'CD56bCD16-', 'CD56dCD16-'): (9, 6),
              ('CD56b', 'CD56dCD16+'): (9, 6),
              ('CD14+CD16+', 'CD14+CD16-', 'CD14low CD16high'): (2, 9), ('D', 'F'): (2, 9),
              ('CD4+', 'CD8+'): (7, 10), ('CD4+', 'CD8+', 'A--', 'CD4+CD8+'): (7, 10),
              ('CD4+HLADR+', ): (4, 5), ('CD8+HLADR+', ): (4, 5),  ('CD4+CD8+HLADR+', ): (4, 5),
              ('T+HLADR+', ): (4, 5), ('CD16-HLADR+', ): (4, 5), ('CD16+HLADR+', ): (4, 5)}
CELL_NAMES = {'CD45+': 'Leukocytes', 'Lymphos': 'Lymphocytes', 'Monos': 'Monocytes', 'Granulos': 'Granulocytes',
              'NKT': 'NKT cells', 'NK': 'NK cells', 'T': 'T cells', 'B': 'B cells', 'Plasma cells': 'Plasma cells',
              'CD56bCD16d': 'CD56bCD16d', 'CD56dCD16+': 'CD56dCD16+', 'CD14+CD16+': 'CD14+CD16+', 'Rest': 'Rest',
              'CD14+CD16-': 'CD14+CD16-', 'CD14low CD16high': 'CD14-CD16+', 'CD4+': 'CD4+ T cells', 'Beads': 'Beads',
              'CD8+': 'CD8+ T cells', 'CD4+HLADR+': 'CD4+ HLA-DR+ T cells', 'CD8+HLADR+': 'CD8+ HLA-DR+ T cells'}
CELL_NAMES_TEX = {'CD45+': 'Leukocytes', 'Lymphos': 'Lymphocytes', 'Monos': 'Monocytes', 'Granulos': 'Granulocytes',
                  'NKT': 'NKT cells', 'NK': 'NK cells', 'T': 'T cells', 'B': 'B cells', 'Plasma cells': 'Plasma cells',
                  'CD56bCD16d': 'CD56^{bright}CD16^{dim}', 'CD56dCD16+': 'CD56^{dim}CD16^+', 'Rest': 'Rest',
                  'CD14+CD16+': 'CD14^+CD16^+', 'CD14+CD16-': 'CD14^+CD16^-', 'CD14low CD16high': 'CD14^-CD16^+',
                  'CD4+': 'CD4^+ T cells', 'CD8+': 'CD8^+ T cells', 'Beads': 'Beads', 'CD56b': 'CD56^{bright}',
                  'CD4+HLADR+': 'CD4^+ HLA-DR^+ T cells', 'CD8+HLADR+': 'CD8^+ HLA-DR^+ T cells'}
SCALING = {'arcsinh': ['SSC', 'FSC', 'CD14-FITC', 'CD138-PE', 'HLA DR-ECD', 'CD3-PC5.5',
                       'CD56-PC7', 'CD4-APC', 'CD19-A700', 'CD16-A750', 'CD8-PB', 'CD45-KrOrange']}
MISSING_F_D_PANEL_FILES = ['AC11041983', 'HS09031994']
SCALING_METHODS = {'arcsinh': lambda x: np.arcsinh(x / 5) - 4}
INV_SCALING_METHODS = {'arcsinh': lambda x: 5 * np.sinh(x + 4)}


def get_gate_panel(kid, pedigree):
    for parent, kids in pedigree:
        if kid in kids:
            return '|'.join(kids)
    return None


def get_ms_files_df(paths: dict, idxs: list, probe: str, gate_panel: str, parent_gate_panel: str = None):
    df = get_files_df(paths['fcs'], paths['labels'], paths['labels'], idxs, substr=f'{probe}0', gate_substr=parent_gate_panel)
    if parent_gate_panel is not None:
        from pathlib import Path
        df['gates'] = df.fcs.apply(lambda x: f'{paths["gates"]}/{Path(x).stem}_{parent_gate_panel}.ftr')
    if gate_panel == 'D|F':
        df = df[~df.fcs.str.contains('|'.join(MISSING_F_D_PANEL_FILES))]
    return df
