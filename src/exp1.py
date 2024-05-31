import pandas as pd
import os
from src.metrics import metrics
import numpy as np
from collections import defaultdict
import yaml

class Experiment():
    def __init__(self, config):
        self.config = config

        # LOG PRINT
        metric_name = self.config['metric']
        db_name = self.config['db']
        print(f'*** METRIC: {metric_name} ***')
        print(f'*** DB: {db_name} ***\n')

        # Import data (quality metrics, rates, mos)g
        self.db_mos = pd.read_csv(self.config['csv_speech'])

        # Filter database
        self.db_mos = self.db_mos[self.db_mos['db'] == self.config['db']]

        # Get rates
        rater_list = [str(x) for x in list(range(0,100))]
        rater_columns = [val for val in self.db_mos.columns if val in rater_list]
        num_raters = len(rater_columns)
        if self.config['per_cond']:
            rater_columns = rater_columns + ['Condition']
        self.raters = self.db_mos[rater_columns]
        
        # Convert to int
        rater_list = range(1, num_raters + 1)
        rater_cols = self.raters.columns.to_list()[:num_raters]
        mapping = {rc:rl for rc, rl in zip(rater_cols, rater_list)}
        self.raters.rename(mapping, axis=1, inplace=True)

        # Group MOS predictions
        if self.config['per_cond']:
            self.db_mos = self.db_mos.groupby('Condition')[[self.config['metric'], 'MOS']].mean()
        
        # CCI fig path
        fig_path = self.config['out_dir'] + metric_name  + '_' + db_name + '.png'

        # Get Population Parameters
        self.pcc_population = metrics.pcc(self.db_mos['MOS'], self.db_mos[self.config['metric']])
        self.srcc_population = metrics.srcc(self.db_mos['MOS'], self.db_mos[self.config['metric']])
        self.ktau_population = metrics.ktau(self.db_mos['MOS'], self.db_mos[self.config['metric']])
        self.cci_population = metrics.cci(self.db_mos[self.config['metric']], self.raters, digits=3, fig_path=fig_path, per_cond=self.config['per_cond'], plot=self.config['plot_cci'])
        self.wcci_population = metrics.wcci(self.db_mos[self.config['metric']], self.raters, digits=3)
        self.pop_df = pd.DataFrame.from_dict({'pcc': self.pcc_population, 'srcc': self.srcc_population, 'ktau': self.ktau_population, 'cci': self.cci_population, 'wcci': self.wcci_population}, orient='index').rename({0: 'population'}, axis=1)
        
        # Array sample sizes (log spaced)
        db_size = len(self.db_mos)
        self.n = np.geomspace(10, db_size - 2, 20).astype(int)
        pass
        # Initialize dictionary to map the namespace of metrics.py functions

    def get_score(self, metric_name, db_mos):
        func = getattr(metrics, metric_name)
        if (metric_name != 'cci') & (metric_name != 'wcci'):
            metric_score = func(db_mos['MOS'], db_mos[self.config['metric']])
        else:
            rater_list = [str(x) for x in list(range(0,100))]
            rater_columns = [val for val in db_mos.columns if val in rater_list]
            raters = db_mos[rater_columns]
            raters.columns = raters.columns.astype(int)
            metric_score = func(db_mos[self.config['metric']], raters, digits=3)

        return metric_score
    
    def compute_metrics(self, db_mos):
        # Performance data store
        scores = defaultdict(list)

        # Calculate each metric set
        for metric_name in self.config['metric_names']:
            metric_score = self.get_score(metric_name, db_mos)

            # Store score
            scores[metric_name] = metric_score
        
        scores = pd.DataFrame.from_dict(scores, orient='index').rename({0: 'score'}, axis=1)
        return scores
    
    def metric_stats(self, metric_scores):
        metric_scores = pd.concat(metric_scores, axis=1)
        metric_scores[metric_scores == 0] = np.nan
        ms_mean = np.nanmean(metric_scores, axis=1)
        ms_std = np.nanstd(metric_scores, axis=1)
        ms_p5 = metric_scores.quantile(q=0.05, axis=1)
        ms_p95 = metric_scores.quantile(q=0.95, axis=1)
        return ms_mean, ms_std, ms_p5, ms_p95

    def samples_loop(self):
        scores_n = defaultdict(list)
        mean_n, std_n, p5_n, p95_n = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)

        for sample_size in self.n:
            print(f'SAMPLE SIZE: {sample_size}', flush=True)

            for id_sample in range(self.config['S']):
                # Sampling data (use id sample to use a different seed for each sample)
                db_mos = self.db_mos.sample(sample_size, random_state=id_sample)
                
                # Compute metrics
                scores = self.compute_metrics(db_mos)

                # Store for each sample size (S times)
                scores_n[sample_size].append(scores)
            
            sample_mean, sample_std, sample_p5, sample_p95 = self.metric_stats(scores_n[sample_size])
            mean_n[sample_size] = sample_mean
            std_n[sample_size] = sample_std
            p5_n[sample_size] = sample_p5
            p95_n[sample_size] = sample_p95

        # Save stats
        mean_n = pd.DataFrame(mean_n)
        self.save_stats_csv(mean_n, 'mean')
        
        std_n = pd.DataFrame(std_n)
        self.save_stats_csv(std_n, 'std')
        
        p5_n = pd.DataFrame(p5_n)
        self.save_stats_csv(p5_n, 'p5_n')
        
        p95_n = pd.DataFrame(p95_n)
        self.save_stats_csv(p95_n, 'p95_n')
        
        # Save config file used for this experiment
        out_path_config = os.path.join(self.config['out_dir'], 'config.yaml') 
        with open(out_path_config, 'w') as out_config_file:
            _ = yaml.dump(self.config, out_config_file)
        pass    

    def save_stats_csv(self, df_stats, stats_name):
        if (stats_name == 'mean') | (stats_name == 'std'):
            stats_cat = pd.concat([df_stats, self.pop_df.reset_index()], axis=1).reset_index().round(3).drop('level_0', axis=1).rename({'index': 'metric'}, axis=1)
        elif (stats_name == 'p5_n') | (stats_name == 'p95_n'):
            stats_cat = pd.concat([df_stats, self.pop_df], axis=1).reset_index().round(3).rename({'index': 'metric'}, axis=1)

        metric = self.config['metric']
        db = self.config['db']
        sub_dir = f'{metric}_{db}'
        out_path = os.path.join(self.config['out_dir'], sub_dir)
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        out_path = os.path.join(out_path, f'{stats_name}.csv')
        stats_cat.to_csv(out_path, index=False)