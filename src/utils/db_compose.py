import pandas as pd
import os

root = '/media/alergn/hdd/datasets/audio/speech'

# EXP 1
exp1_acr = os.path.join(root, 'ITU-T P_Suppl_23_DB/exp1/subjective_scores.csv')
exp1_path = os.path.join(root, 'ITU-T P_Suppl_23_DB/exp1/visqol_format.csv')
exp1_cond = os.path.join(root, 'ITU-T P_Suppl_23_DB/P_Suppl_23_DB/Disk1/mos_ratings.csv')

exp1_acr_df = pd.read_csv(exp1_acr)
exp1_path_df = pd.read_csv(exp1_path)
exp1_path_df['Filename'] = [x.split('/')[-1] for x in exp1_path_df['degraded']]
exp1_cond_df = pd.read_csv(exp1_cond)

db_exp1 = exp1_cond_df.merge(exp1_acr_df, on='Filename').merge(exp1_path_df, on='Filename')

# EXP 3
exp3_acr = os.path.join(root, 'ITU-T P_Suppl_23_DB/exp3/subjective_scores.csv')
exp3_path = os.path.join(root, 'ITU-T P_Suppl_23_DB/exp3/visqol_format.csv')
exp3_cond = os.path.join(root, 'ITU-T P_Suppl_23_DB/P_Suppl_23_DB/Disk3/mos_ratings.csv')

exp3_acr_df = pd.read_csv(exp3_acr)
exp3_path_df = pd.read_csv(exp3_path)
exp3_path_df['Filename'] = [x.split('/')[-1] for x in exp3_path_df['degraded']]
exp3_cond_df = pd.read_csv(exp3_cond)

db_exp3 = exp3_cond_df.merge(exp3_acr_df, on='Filename').merge(exp3_path_df, on='Filename')

# TCD VOIP
visqol_acr_cond = os.path.join(root, 'TCD-VOIP/subjective_scores_cond.csv')
visqol_path = os.path.join(root, 'TCD-VOIP/scores_visqol_format.csv')

visqol_acr_cond_df = pd.read_csv(visqol_acr_cond)
visqol_path_df = pd.read_csv(visqol_path)
visqol_path_df['Filename'] = [x.split('/')[-1] for x in visqol_path_df['degraded']]

db_visqol = visqol_acr_cond_df.merge(visqol_path_df, on='Filename')
db_visqol.drop(['ConditionID', 'Speaker'], axis=1, inplace=True)
db_visqol.rename({'DegCond': 'Condition'}, axis=1, inplace=True)
id_acr = [str(i) for i in list(range(1,25))]
db_visqol['mos'] = db_visqol.loc[:, id_acr].mean(axis=1)


# Add Genspeech
wissam_data = pd.read_csv('/media/alergn/hdd/datasets/audio/speech/ITU-T P_Suppl_23_DB/All_Speech_Data_List_Wissam_New_POLQA.csv')
genspeech_df = wissam_data[wissam_data['DataBase'] == 'Genspeech']
genspeech_df = genspeech_df[['Ref_Wave', 'Test_Wave', 'Codec', 'sampleMOS']]
genspeech_df.rename({'Ref_Wave': 'reference', 'Test_Wave': 'degraded', 'sampleMOS': 'mos', 'Codec': 'Condition'}, axis=1, inplace=True)
genspeech_df['reference'] = [os.path.join('/media/alergn/hdd/datasets/audio/speech', '/'.join(x.split('/')[1:])) for x in genspeech_df['reference']]
genspeech_df['degraded'] = [os.path.join('/media/alergn/hdd/datasets/audio/speech', '/'.join(x.split('/')[1:])) for x in genspeech_df['degraded']]
genspeech_df['Filename'] = [x + '_' + str(id) for id, x in enumerate(genspeech_df['Condition'])]

db = pd.concat([db_exp1, db_exp3, db_visqol, genspeech_df], keys=['P23_EXP1', 'P23_EXP3', 'TCD-VOIP', 'Genspeech']).reset_index().drop('level_1', axis=1).rename({'level_0': 'db'}, axis=1)
db.rename({'reference': 'Filepath Ref', 'degraded': 'Filepath Deg', 'mos': 'MOS'}, axis=1, inplace=True)
pass