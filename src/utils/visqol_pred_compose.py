import pandas as pd
import os

root = '/media/alergn/hdd/datasets/audio/speech'

path_tcd_voip = os.path.join(root, 'TCD-VOIP/visqol_results_file.csv')
df_tcd = pd.read_csv(path_tcd_voip)

path_exp1 = os.path.join(root, 'ITU-T P_Suppl_23_DB/exp1/visqol_results.csv')
df_exp1 = pd.read_csv(path_exp1)
df_exp1.rename({'reference': 'Filepath Ref', 'degraded': 'Filepath Deg'}, axis=1, inplace=True)
df_exp1['Filename'] = [x.split('/')[-1] for x in df_exp1['Filepath Deg']]
df_exp1 = df_exp1[['Filename', 'moslqo']]

path_exp3 = os.path.join(root, 'ITU-T P_Suppl_23_DB/exp3/visqol_results.csv')
df_exp3 = pd.read_csv(path_exp3)
df_exp3.rename({'reference': 'Filepath Ref', 'degraded': 'Filepath Deg'}, axis=1, inplace=True)
df_exp3['Filename'] = [x.split('/')[-1] for x in df_exp3['Filepath Deg']]
df_exp3 = df_exp3[['Filename', 'moslqo']]

visqol_db = pd.concat([df_tcd, df_exp1, df_exp3], axis=0, keys=['TCD-VOIP', 'P23_EXP1', 'P23_EXP3']).reset_index().drop('level_1', axis=1).rename({'level_0': 'db'}, axis=1)
pass