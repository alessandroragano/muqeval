import os
import pandas as pd

db = pd.read_csv('data/test_db.csv')
db = db.rename({'Filepath Deg': 'filename'}, axis=1, inplace=False)
db['filename'].to_csv('data/db_nomad.csv', index=False)

# Librispeech dev set as non-matching references
# root_data = '/media/alergn/hdd/datasets/audio/speech/LibriSpeech/dev-clean'
# ls = pd.read_csv('/media/alergn/hdd/datasets/audio/speech/LibriSpeech/dev-clean.csv')
# ls['filename'] = [os.path.join(root_data, fp, fn) for fn, fp in zip(ls['filename'], ls['filepath'])]
# ls.drop('filepath', axis=1, inplace=True)
pass