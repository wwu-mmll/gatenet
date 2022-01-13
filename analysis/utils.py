LABEL_COL = 0


def flowcap_paths(dataset: str, flowcap_dir: str):
    fcs_path = f'{flowcap_dir}/Data/FCM/fcs/{dataset}/FCS'
    labels_path = f'{flowcap_dir}/Data/Labels/{dataset}'
    return fcs_path, labels_path
