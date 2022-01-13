import numpy as np
import pandas as pd


old = pd.read_csv('final_pseudo.csv')
new = old.copy(deep=True)
sus = pd.read_csv('final_pseudo_sus.csv')
sus = sus[~sus.New_File_Name_New.isna()]

rename_cols = [c for c in old.columns if c not in ['File_Name', 'FACS_Count', 'Event_Count']]

for i in sus.index:
    for c in rename_cols:
        new_postfix = '.1' if 'Letter' in c else '_New'
        new_value = sus.loc[i, c + new_postfix]
        if not pd.isnull(new_value):
            new.loc[i, c] = sus.loc[i, c + new_postfix]

new['harvest_volume'] = new['harvest_volume'].replace({'-': np.nan})

hv = pd.read_csv('add_harvest_volumes.csv')
hv['File_Name'] = hv.File_Name.str.strip()
hv = hv[~hv.Volumen.isna()]

for fn, vol in zip(hv.File_Name, hv.Volumen):
    new.loc[new.File_Name == fn, 'harvest_volume'] = vol

new.to_csv('final_pseudo_new.csv')
