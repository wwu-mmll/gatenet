import numpy as np
LABEL_COL = 0
DATASET_NAME_MAP = {'NDD': 'ND', 'CSFE': 'WNV', 'GvHD': 'GvHD', 'Lymph': 'DLBCL', 'StemCell': 'HSCT'}
FCS_SCALING = {'NDD': {'forward': {'FSC-A': lambda x: np.log(x + 1), 'SSC-A': lambda x: np.log(x + 1)},
                       'backward': {'FSC-A': lambda x: np.exp(x) - 1, 'SSC-A': lambda x: np.exp(x) - 1}}}
FCS_STATS = {'NDD': {'FSC-A': (12., .43), 'SSC-A': (11., .79), 'FITC-A': (1.8, 0.44), 'PerCP-Cy5-5-A': (0.96, 0.34),
                     'Pacific Blue-A': (1.6, 0.36), 'Pacifc Orange-A': (1.3, 0.25), 'QDot 605-A': (0.72, 0.34),
                     'APC-A': (1.1, 0.41), 'Alexa 700-A': (2.4, 0.5), 'PE-A': (1.4, 0.26),
                     'PE-Cy5-A': (2.4, 0.36), 'PE-Cy7-A': (2.2, 0.47)}}


def flowcap_paths(dataset: str, flowcap_dir: str):
    fcs_path = f'{flowcap_dir}/Data/FCM/fcs/{dataset}/FCS'
    labels_path = f'{flowcap_dir}/Data/Labels/{dataset}'
    return fcs_path, labels_path
