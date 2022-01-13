from src.data.files import get_files_df
PEDIGREE = ((None, ['CD45+']),
            ('CD45+', ['Granulos', 'Monos', 'Lymphos']),
            ('Lymphos', ['NKT', 'NK', 'T']),
            ('Lymphos', ['B', 'Plasma cells']),
            ('NK', ['CD56bCD16d', 'CD56dCD16+']),
            (None, ['Beads']),
            ('Monos', ['CD14+CD16+', 'CD14+CD16-', 'CD14low CD16high']),
            ('T', ['CD4+', 'CD8+']),
            ('CD4+', ['CD4+HLADR+']),
            ('CD8+', ['CD8+HLADR+']))
PEDIGREE_TOTAL = ((None, ['CD45+']),
                  (None, ['Beads']),
                  ('CD45+', ['Granulos', 'Monos', 'Lymphos']),
                  ('Lymphos', ['NKT', 'NK', 'T']),
                  ('Lymphos', ['B', 'Plasma cells']),
                  ('Lymphos', ['B-_T-cell doubletes']),
                  ('NK', ['CD56bCD16d', 'CD56dCD16+', 'CD56bCD16-', 'CD56dCD16-']),
                  ('Monos', ['CD14+CD16+', 'CD14+CD16-', 'CD14low CD16high']),
                  ('Monos', ['D', 'F']),
                  ('T', ['CD4+', 'CD8+', 'A--', 'CD4+CD8+']),
                  ('CD4+', ['CD4+HLADR+']),
                  ('CD8+', ['CD8+HLADR+']),
                  ('CD4+CD8+', ['CD4+CD8+HLADR+']),
                  ('T', ['T+HLADR+']),
                  ('D', ['CD16-HLADR+']),
                  ('F', ['CD16+HLADR+']))
PEDIGREES = {'small': PEDIGREE, 'total': PEDIGREE_TOTAL}
NO_REST_PANELS = ['CD4+_CD8+_A--_CD4+CD8+', 'CD56bCD16d_CD56dCD16+_CD56bCD16-_CD56dCD16-']
MARKERS = {('CD45+', ): ['CD45-KrOrange', 'FSC'],
           ('Beads', ): ['TIME', 'HLA DR-ECD'],
           ('Granulos', 'Monos', 'Lymphos'): ['CD14-FITC', 'SSC'],
           ('NKT', 'NK', 'T'): ['CD3-PC5.5', 'CD56-PC7'],
           ('B', 'Plasma cells'): ['CD138-PE', 'CD19-A700'],
           ('CD56bCD16d', 'CD56dCD16+'): ['CD16-A750', 'CD56-PC7'],
           ('CD14+CD16+', 'CD14+CD16-', 'CD14low CD16high'): ['CD14-FITC', 'CD16-A750'],
           ('CD4+', 'CD8+'): ['CD4-APC', 'CD8-PB'],
           ('CD4+HLADR+', ): ['HLA DR-ECD', 'CD3-PC5.5'],
           ('CD8+HLADR+', ): ['HLA DR-ECD', 'CD3-PC5.5']}
FCS_COL_NAMES = ['FSC', 'SSC', 'CD14-FITC', 'CD138-PE', 'HLA DR-ECD', 'CD3-PC5.5', 'CD56-PC7',
                 'CD4-APC', 'CD19-A700', 'CD16-A750', 'CD8-PB', 'CD45-KrOrange', 'TIME']
CELL_NAMES = {'CD45+': 'Leukocytes', 'Lymphos': 'Lymphocytes', 'Monos': 'Monocytes', 'Granulos': 'Granulocytes',
              'NKT': 'NKT cells', 'NK': 'NK cells', 'T': 'T cells', 'B': 'B cells', 'Plasma cells': 'Plasma cells',
              'CD56bCD16d': 'CD56bCD16d', 'CD56dCD16+': 'CD56dCD16+', 'CD14+CD16+': 'CD14+CD16+', 'Rest': 'Rest',
              'CD14+CD16-': 'CD14+CD16-', 'CD14low CD16high': 'CD14-CD16+', 'CD4+': 'CD4+ T cells', 'Beads': 'Beads',
              'CD8+': 'CD8+ T cells', 'CD4+HLADR+': 'CD4+ HLA-DR+ T cells', 'CD8+HLADR+': 'CD8+ HLA-DR+ T cells'}
SCALING = {'arcsinh': ['CD14-FITC', 'CD138-PE', 'HLA DR-ECD', 'CD3-PC5.5', 'CD56-PC7',
                       'CD4-APC', 'CD19-A700', 'CD16-A750', 'CD8-PB', 'CD45-KrOrange'],
           'div_1000': ['SSC', 'FSC']}
MISSING_F_D_PANEL_FILES = ['AC11041983', 'HS09031994']


def get_gate_panel(kid, pedigree):
    for parent, kids in pedigree:
        if kid in kids:
            return '_'.join(kids)
    return None


def get_ms_files_df(paths: dict, idxs: list, probe: str, gate_panel: str, parent_gate_panel: str = None):
    df = get_files_df(paths['fcs'], paths['labels'], None, idxs, substr=f'{probe}0', gate_substr=parent_gate_panel)
    if parent_gate_panel is not None:
        from pathlib import Path
        df['gates'] = df.fcs.apply(lambda x: f'{paths["gates"]}/{Path(x).stem}_{parent_gate_panel}.csv')
    if gate_panel == 'D_F':
        df = df[~df.fcs.str.contains('|'.join(MISSING_F_D_PANEL_FILES))]
    return df


def get_batch_idxs(n_events, max_count=2 * 10**6, skip_i=None):
    all_idxs = []
    batch_idxs = []
    n_sum = 0
    for i, n in enumerate(n_events):
        if i not in skip_i:
            n_sum += n
            if n_sum > max_count:
                n_sum = n
                all_idxs.append(batch_idxs)
                batch_idxs = []
            else:
                batch_idxs.append(i)
    return all_idxs
