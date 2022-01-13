import pandas as pd
import os, shutil
from glob import glob
from pathlib import Path


old_fp = '/mnt/data-ssd/cyto/lmd_11k_backup'
new_fp = '/mnt/data-ssd/cyto/lmd_11k_backup_changed'
chng = pd.read_csv('pseudo_change.csv', index_col=[0])

for idx, row in chng.iterrows():
    shutil.copy(f'{old_fp}/{row.Old_File_Name}', f'{new_fp}/{row.New_File_Name}')


dest_dir = '/mnt/data-ssd/cyto/lmd_11k'

for idx, row in chng.iterrows():
    os.remove(f'{dest_dir}/{row.Old_File_Name}')


for f in glob('/mnt/data-ssd/cyto/lmd_11k_backup_changed/*'):
    shutil.copy(f, f'{dest_dir}/{Path(f).name}')
