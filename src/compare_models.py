import pandas as pd
import os
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set(font_scale=2.2)
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

class Experiment():
    def __init__(self, config):
        
        sample_means = []
        model_name = []
        db_name = []
        self.config = config
        for model in config['metric']:
            for db in config['db']:
                model_db_name = f'{model}_{db}'
                smean_path = os.path.join(config['stats_root'], config['exp'], model_db_name, 'mean.csv')
                sm_df = pd.read_csv(smean_path)
                model_name.append([model]*len(sm_df))
                db_name.append([db]*len(sm_df))
                sample_means.append(sm_df)
        
        sample_means_df = pd.concat(sample_means)
        sample_means_df['model'] = np.concatenate(model_name)
        sample_means_df['db'] = np.concatenate(db_name)
        sample_means_df = sample_means_df[sample_means_df['metric'].isin(config['metric_names'])]

        def relative_distance(row, population_param, use_abs=True):
            if use_abs:
                return pd.Series([100*abs((x - population_param)/population_param) for x in row])
            else:
                return pd.Series([100*(x - population_param)/population_param for x in row])
        
        def metric_distance(df):
            if 'Perc Point' in df:
                perc_point = df['Perc Point']
                df.drop('Perc Point', axis=1, inplace=True)
                abs = False
            else:
                perc_point = None
                abs = True
            distance = df.set_index(['metric', 'model', 'db']).apply(lambda row: relative_distance(row, row['population'], abs), axis=1)
            distance.columns = df.drop(['metric', 'model', 'db'], axis=1).columns
            distance.drop('population', axis=1, inplace=True)

            if perc_point is not None:
                distance['Perc Point'] = perc_point.values
            return distance

        color_variables = {
            'CCI': 'blue',
            'WCCI': 'brown',
            'PCC': 'red',
            'SRCC': 'green',
            'KTAU': 'black'
        }

        #rel_dist = metric_distance(sample_means_df)
        rel_dist = sample_means_df
        col_names = sample_means_df.columns[:-3]
        df_melt = pd.melt(rel_dist.reset_index(), id_vars=['metric', 'model', 'db'], value_vars=col_names, ignore_index=False, var_name='x_axis', value_name='y_axis')
        df_melt['metric'] =  df_melt['metric'].str.upper()
        db_grouped = df_melt.groupby('db')
        linestyles = {'PESQ': 'solid', 'VISQOL': 'dotted'}
        markers = {'PESQ': 'o', 'VISQOL': 'v'}
        for dbg_name, dbg in db_grouped:
            plt.figure(figsize=(10, 10))
            g = sns.lineplot(data=dbg, x='x_axis', y='y_axis', hue='metric', style='model', palette=color_variables, markers=True)
            g.legend_.set_title(None)
            plt.gca().xaxis.grid(True)
            plt.xlabel('Raters')
            plt.ylabel(r'$|\frac{\hat{\rho} - \rho}{\rho}|(\%)$')
            plt.xticks(rotation=90)
            plt.tight_layout()            
            plt.savefig(os.path.join(config['stats_root'], config['exp'], f'figs/compare_{dbg_name}.pdf'), dpi=150)
            plt.close()
        pass