import pandas as pd
import os
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale=1.6)
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":", 'legend.frameon':True})

class Experiment():
    def __init__(self, config):
        
        self.config = config

        # # LOG PRINT
        # metric_name = self.config['metric']
        # db_name = self.config['db']
        # print(f'*** METRIC: {metric_name} ***')
        # print(f'*** DB: {db_name} ***\n')

        # Import statistics metric
        #stats_path = os.path.join(config['stats_root'], config['exp'], metric_name + '_' + db_name)
        stats_path = os.path.join(config['stats_root'], config['exp'])
        rr2_name = 'rr_2.csv'
        rr4_name = 'rr_4.csv'
        if config['metric'] == 'ssim':
            rr2_name = 'rr_2_ssim.csv'
            rr4_name = 'rr_4_ssim.csv'
        rr2 = pd.read_csv(os.path.join(stats_path, rr2_name)).rename({'index': 'metric'}, axis=1)
        rr2 = rr2[rr2['metric'].isin(config['metric_names'])]
        rr4 = pd.read_csv(os.path.join(stats_path, rr4_name)).rename({'index': 'metric'}, axis=1)
        rr4 = rr4[rr4['metric'].isin(config['metric_names'])]

        def relative_distance(row, population_param):
            return pd.Series([abs(x - population_param) for x in row])
        
        def metric_distance(df):
            if 'Model DB' in df.columns:
                index = ['metric', 'Model DB']
            else:
                index = ['metric']
            distance = df.set_index(index).apply(lambda row: relative_distance(row, row['population']), axis=1)
            distance.columns = df.set_index(index).columns
            distance.drop('population', axis=1, inplace=True)
            return distance

        plot_variables = defaultdict(list)
        plot_variables['rr2'] = metric_distance(rr2).reset_index()
        plot_variables['rr4'] = metric_distance(rr4).reset_index()
        #plot_variables['p5_distance'] = metric_distance(p5_df)
        #plot_variables['p95_distance'] = metric_distance(p95_df)

        # color_variables = {
        #     'CCI': 'blue',
        #     'WCCI': 'brown',
        #     'PCC': 'red',
        #     'SRCC': 'green',
        #     'KTAU': 'black'
        # }

        if self.config['metric'] == 'ssim':
            # color_variables = {
            #     'SSIM_JPEGXR': 'blue',
            #     'SSIM_JPEGXR': 'orange',
            #     'SSIM_JPEGXR': 'red'
            # }
            color_variables = None
        else:
            color_variables = {
                'VISQOL_P23_EXP1': 'blue',
                'VISQOL_P23_EXP3': 'orange',
                'VISQOL_TCD-VOIP': 'red',
                'PESQ_P23_EXP1': 'lightgreen',
                'PESQ_P23_EXP3': 'black',
                'PESQ_TCD-VOIP': 'gray'
            }
        
        #'p5_distance': r'$|\frac{\hat{\rho}_{5} - \rho}{\rho}|(\%)$',
        #'p95_distance': r'$|\frac{\hat{\rho}_{95} - \rho}{\rho}|(\%)$',
        
        ylabel_mapping = {
            'mean_distance': r'$|\frac{\hat{\rho} - \rho}{\rho}|(\%)$',
            'relative_std': r'$c_{v}(\%)$',
            'perc_point_distance': r'$|\frac{\hat{\rho}_{5,95} - \rho}{\rho}|(\%)$'
        }

        # Plot Labels
        xlabel = 'Restricted Range'


        for name_var, var in plot_variables.items():
            if 'Model DB' in var.columns:
                index = ['metric', 'Model DB']
            else:
                index = ['metric']
            # Unpivoting metric statistics
            var_unpivoted = pd.melt(var, id_vars=index, value_vars=var.columns, var_name='Range', value_name='Distance')
            var_unpivoted['metric'] = var_unpivoted['metric'].str.upper()
            
            # Group by Range
            range_group = var_unpivoted.groupby('Range')

            for range_name, range_vals in range_group:
                # Plot
                plt.figure(figsize=(20, 20))
                if 'Model DB' in range_vals:
                    hue = 'Model DB'
                else:
                    hue = None
                g = sns.catplot(data=range_vals, x='metric', y='Distance', hue=hue, kind='swarm', palette=color_variables, s=100)
                #if (name_var != 'rr4') | (range_name!='Excellent'): 
                if g.legend:
                    g._legend.remove()
                #else:
                #    sns.move_legend(g, "upper right", bbox_to_anchor=(0.9, 1.1))

                # Plot modify
                plt.gca().xaxis.grid(True)
                plt.xlabel('')
                plt.ylabel(r'$|\hat{\rho} - \rho|$')
                plt.xticks(rotation=90)
                #if (name_var != 'rr4') | (range_name!='Excellent'): 
                plt.tight_layout()
                path_fig = f'{stats_path}/figs'
                if not os.path.isdir(path_fig):
                    os.makedirs(path_fig)
                plt.savefig(f'{path_fig}/{name_var}_{range_name}.png', dpi=200)
                plt.close()