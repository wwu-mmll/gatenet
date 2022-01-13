import fcsparser
from fastai.data.all import pd, plt
import seaborn as sn


# fcs_path = '/mnt/data-ssd/cyto/fastcyto_dirs/extracted_stuff_big/lmds/BL06061989L001.LMD'
# labels_path = '/mnt/data-ssd/cyto/fastcyto_dirs/extracted_stuff_big/labels/BL06061989L001.csv'

fcs_path = '/spm-data/Scratch/spielwiese_lukas/data/cyto/lmd_11k/00001-00-L001-1.LMD'
labels_path = '/home/lfisch/Projects/gatenet/data/predictions/ms/pedigree_total/L/argmax_preds/00001-00-L001-1.csv'

fcs = fcsparser.parse(fcs_path, reformat_meta=True)[1]
label = pd.read_csv(labels_path)
df = pd.concat([fcs, label], axis=1)
df = df[df['T'] == 1]

sn.scatterplot(data=df, x='HLA DR-ECD', y='CD3-PC5.5', hue='T+HLADR+')
plt.show()
