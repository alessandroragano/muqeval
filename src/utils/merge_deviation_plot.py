import pandas as pd
import os

stats = ['rr_2.csv', 'rr_4.csv']
path = './experiments/exp3'

rr2_df = []
rr4_df = []
metric_db_rr2 = []
metric_db_rr4 = []

for root, dir, files in os.walk(path):
    for f in files:
        if f in stats:
            path_file = pd.read_csv(os.path.join(root, f))
            metric_db_name = root.split('/')[-1]
            if f == 'rr_2.csv':
                rr2_df.append(path_file)
                metric_db_rr2.append(metric_db_name)
            elif f == 'rr_4.csv':
                rr4_df.append(path_file)
                metric_db_rr4.append(metric_db_name)
metric_db_rr2 = metric_db_rr2*len(path_file)
metric_db_rr4 = metric_db_rr4*len(path_file)

rr2_df = pd.concat(rr2_df)
rr2_df['Model DB'] = metric_db_rr2
rr2_df.rename({'index': 'metric'}, axis=1, inplace=True)
rr2_df.to_csv(os.path.join(path, 'rr_2.csv'), index=False)
rr4_df = pd.concat(rr4_df)
rr4_df['Model DB'] = metric_db_rr4
rr4_df.rename({'index': 'metric'}, axis=1, inplace=True)
rr4_df.to_csv(os.path.join(path, 'rr_4.csv'), index=False)
pass