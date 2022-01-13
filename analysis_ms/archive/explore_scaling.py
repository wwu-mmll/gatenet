import fcsparser
from fastai.data.all import np, pd, plt
import seaborn as sn


fcs_path = '/mnt/data-ssd/cyto/fastcyto_dirs/extracted_stuff_big/lmds/BL06061989L001.LMD'
labels_path = '/mnt/data-ssd/cyto/fastcyto_dirs/extracted_stuff_big/labels/BL06061989L001.csv'
label = pd.read_csv(labels_path)

fcs = fcsparser.parse(fcs_path, reformat_meta=True)[1]
fcs = fcs[label.Beads == 0]
fcs_sc = fcs.apply(lambda x: np.arcsinh(x / 5))

for c in fcs.columns.tolist()[:-1]:
    f, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0] = sn.histplot(fcs[c].values, ax=axs[0])
    axs[1] = sn.histplot(fcs_sc[c].values, ax=axs[1])
    plt.title(c)
    plt.savefig(f'{c}.png')
