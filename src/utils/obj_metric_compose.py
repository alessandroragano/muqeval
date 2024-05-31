import pandas as pd
from scipy.optimize import curve_fit
import numpy as np

def objective(x, a, b):
 return a * x + b

path_visqol = 'VISQOL_results.csv'
df_visqol = pd.read_csv(path_visqol)
path_nisqa = 'NISQA_results.csv'
df_nisqa = pd.read_csv(path_nisqa)
path_pesq = 'PESQ_results.csv'
df_pesq = pd.read_csv(path_pesq)
df_pesq['Filename'] = [x.split('/')[-1] for x in df_pesq['Filepath Deg']]

df = df_pesq.merge(df_visqol, on='Filename').merge(df_nisqa, on='Filename')
df.drop(['db_y', 'model', 'noi_pred', 'dis_pred', 'col_pred', 'loud_pred', 'Filepath Deg_y', 'Unnamed: 0'], axis=1, inplace=True)
df.rename({'mos_pred': 'NISQA', 'moslqo': 'VISQOL', 'db_x': 'db', 'Match Distance': 'PESQ', 'Filepath Deg_x': 'Filepath Deg'}, axis=1, inplace=True)
df.drop_duplicates(inplace=True)
pass

# # ** NOMAD EXCLUDED BECAUSE CHOPPED SPEECH PERFORMANCE ARE BAD **
# path_nomad = '/home/alergn/Documents/testnomad/results-csv/12-10-2023_14-54-17/12-10-2023_14-54-17_nomad_avg.csv'
# df_nomad = pd.read_csv(path_nomad)
# df_nomad.rename({'Test File': 'Filename'}, axis=1, inplace=True)
# df_nomad['Filename'] = [x + '.wav' for x in df_nomad['Filename']]

# df = df_nisqa.merge(df_visqol, on='Filename').merge(df_nomad, on='Filename')
# df.drop_duplicates(inplace=True)

# df.drop(['db_y', 'model', 'noi_pred', 'dis_pred', 'col_pred', 'loud_pred'], axis=1, inplace=True)
# df.rename({'mos_pred': 'NISQA', 'moslqo': 'VISQOL', 'db_x': 'db'}, axis=1, inplace=True)

# df_g = df.groupby('db')

# nomad_map = []
# nomad_map_filenames = []

# for db_type in df_g:
#     popt, _ = curve_fit(objective, db_type[1]['NOMAD'], db_type[1]['MOS'])
#     a, b = popt
#     nomad_map.append(objective(db_type[1]['NOMAD'], a, b))
#     nomad_map_filenames.append(db_type[1]['Filename'].to_list())

# nomad_map = np.concatenate(nomad_map)
# nomad_map_filenames = np.concatenate(nomad_map_filenames)
# df_nomad_map = pd.DataFrame({'Filename': nomad_map_filenames, 'NOMAD_MAP': nomad_map})
# df = df.merge(df_nomad_map, on='Filename')

pass