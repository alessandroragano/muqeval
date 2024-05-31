import pandas as pd
import os
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale=2.2)
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

class Experiment():
    def __init__(self, config):
        
        self.config = config

        # LOG PRINT
        metric_name = self.config['metric']
        db_name = self.config['db']
        print(f'*** METRIC: {metric_name} ***')
        print(f'*** DB: {db_name} ***\n')

        # Import statistics metric
        if config['aggr']:
            stats_path = os.path.join(config['stats_root'], config['exp'])
        else:
            stats_path = os.path.join(config['stats_root'], config['exp'], metric_name + '_' + db_name)
        mean_df = pd.read_csv(os.path.join(stats_path, 'mean.csv'))
        mean_df = mean_df[mean_df['metric'].isin(config['metric_names'])]
        std_df = pd.read_csv(os.path.join(stats_path, 'std.csv'))
        std_df = std_df[std_df['metric'].isin(config['metric_names'])]
        p5_df = pd.read_csv(os.path.join(stats_path, 'p5_n.csv'))
        p5_df = p5_df[p5_df['metric'].isin(config['metric_names'])]
        p95_df = pd.read_csv(os.path.join(stats_path, 'p95_n.csv'))
        p95_df = p95_df[p95_df['metric'].isin(config['metric_names'])]
        perc_point_df = pd.concat([p5_df, p95_df], keys=['p5', 'p95']).reset_index().drop('level_1', axis=1).rename({'level_0': 'Perc Point'}, axis=1)
        
        if self.config['aggr']:
            if config['exp'] == 'exp1':
                col_names = [str(x) for x in list(range(1,21))]
            elif config['exp'] == 'exp2':
                col_names = [str(x) for x in list(range(12,21))]
        else:
            col_names = None

        def relative_distance(row, population_param, use_abs=True):
            if use_abs:
                return pd.Series([abs((x - population_param)/1) for x in row])
            else:
                return pd.Series([(x - population_param)/1 for x in row])
        
        def metric_distance(df):
            if 'Perc Point' in df:
                perc_point = df['Perc Point']
                df.drop('Perc Point', axis=1, inplace=True)
                abs = False
            else:
                perc_point = None
                abs = True
            distance = df.set_index('metric').apply(lambda row: relative_distance(row, row['population'], abs), axis=1)
            distance.columns = df.drop('metric', axis=1).columns
            distance.drop('population', axis=1, inplace=True)

            if perc_point is not None:
                distance['Perc Point'] = perc_point.values
            return distance

        plot_variables = defaultdict(list)
        plot_variables['mean_distance'] = metric_distance(mean_df)
        plot_variables['perc_point_distance'] = metric_distance(perc_point_df)
        #plot_variables['p5_distance'] = metric_distance(p5_df)
        #plot_variables['p95_distance'] = metric_distance(p95_df)
        #relative_std = 100*std_df.drop(['metric', 'population'], axis=1).div(mean_df.drop(['metric', 'population'], axis=1))
        std = std_df.drop(['metric', 'population'], axis=1)
        std['metric'] = std_df['metric']
        plot_variables['std'] = std

        color_variables = {
            'CCI': 'blue',
            'WCCI': 'brown',
            'PCC': 'red',
            'SRCC': 'green',
            'KTAU': 'black'
        }
        #'p5_distance': r'$|\frac{\hat{\rho}_{5} - \rho}{\rho}|(\%)$',
        #'p95_distance': r'$|\frac{\hat{\rho}_{95} - \rho}{\rho}|(\%)$',
        
        ylabel_mapping = {
            'mean_distance': r'$|\hat{\rho} - \rho|$',
            'std': r'$\sigma$',
            'perc_point_distance': r'$|\hat{\rho}_{5,95} - \rho|$'
        }

        # Plot Labels
        if config['exp'] == 'exp1':
            xlabel = 'Sample Size'
        elif config['exp'] == 'exp2':
            xlabel = 'Raters'

        for name_var, var in plot_variables.items():
            # Unpivoting metric statistics
            var.reset_index(inplace=True)
            id_vars = ['metric']
            if name_var == 'perc_point_distance':
                id_vars.append('Perc Point')
            
            if col_names == None:
                col_names = var.columns.to_list()[1:]
            var_unpivoted = pd.melt(var, id_vars=id_vars, value_vars=col_names, ignore_index=False, var_name='x_axis', value_name='y_axis')
            var_unpivoted['metric'] = var_unpivoted['metric'].str.upper()

            # Plot
            plt.figure(figsize=(10, 10))
            if 'Perc Point' in var_unpivoted.columns:
                p_group = var_unpivoted.groupby('Perc Point')
                linestyles = {'p5': 'solid', 'p95': 'dotted'}
                markers = {'p5': 'o', 'p95': 'v'}
                color_variables_percpoint = {'CCI_p5': 'blue', 'WCCI_p5': 'brown', 'PCC_p5': 'red', 'SRCC_p5': 'green','KTAU_p5': 'black',
                                            'CCI_p95': 'blue', 'WCCI_p95': 'brown', 'PCC_p95': 'red', 'SRCC_p95': 'green','KTAU_p95': 'black'}
                for p_name, p_val in p_group:
                    p_val['metric'] = [x + '_' + y for x,y in zip(p_val['metric'], p_val['Perc Point'])]
                    
                    g = sns.pointplot(data=p_val, x='x_axis', y='y_axis', hue='metric', palette=color_variables_percpoint, markers=markers[p_name], linestyles=linestyles[p_name], scale=1.5)
                sns.move_legend(g, "lower right", bbox_to_anchor=(1.1, -0.04), ncol=2)
            else:
                g = sns.pointplot(data=var_unpivoted, x='x_axis', y='y_axis', hue='metric', palette=color_variables, markers='o', scale=1.5)
            
            # Plot modify
            g.legend_.set_title(None)
            plt.gca().xaxis.grid(True)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel_mapping[name_var])
            plt.xticks(rotation=90)
            plt.tight_layout()
            path_fig = f'{stats_path}/figs'
            if not os.path.isdir(path_fig):
                os.makedirs(path_fig)
            plt.savefig(f'{path_fig}/{name_var}.png', dpi=200)
            plt.close()