import pandas as pd
import os
from src.metrics import metrics
import numpy as np
from collections import defaultdict
import yaml
import random

class Experiment():
    def __init__(self, config):
        
        self.config = config

        # LOG PRINT
        metric_name = self.config['metric']
        db_name = self.config['db']
        print(f'*** METRIC: {metric_name} ***')
        print(f'*** DB: {db_name} ***\n')

        # Import data (quality metrics, rates, mos)
        self.db_mos = pd.read_csv(self.config['csv_speech'])    

        # Filter database
        self.db_mos = self.db_mos[self.db_mos['db'] == self.config['db']]

        # Get rates
        rater_list = [str(x) for x in list(range(0,100))]
        rater_columns = [val for val in self.db_mos.columns if val in rater_list]
        self.raters = self.db_mos[rater_columns]
        self.raters.columns = self.raters.columns.astype(int)

        # Get Population Parameters
        self.pcc_population = metrics.pcc(self.db_mos['MOS'], self.db_mos[self.config['metric']])
        self.srcc_population = metrics.srcc(self.db_mos['MOS'], self.db_mos[self.config['metric']])
        self.ktau_population = metrics.ktau(self.db_mos['MOS'], self.db_mos[self.config['metric']])
        self.cci_population = metrics.cci(self.db_mos[self.config['metric']], self.raters, digits=3)
        self.wcci_population = metrics.wcci(self.db_mos[self.config['metric']], self.raters, digits=3)
        self.pop_df = pd.DataFrame.from_dict({'pcc': self.pcc_population, 'srcc': self.srcc_population, 'ktau': self.ktau_population, 'cci': self.cci_population, 'wcci': self.wcci_population}, orient='index').rename({0: 'population'}, axis=1)
        
        # Array number of participants
        self.m = list(range(config['min'], config['max']+1))

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
        #ms_p5 = np.percentile(metric_scores, 5)
        #ms_p95 = np.percentile(metric_scores, 95)
        ms_p5 = metric_scores.quantile(q=0.05, axis=1)
        ms_p95 = metric_scores.quantile(q=0.95, axis=1)
        return ms_mean, ms_std, ms_p5, ms_p95

    def samples_loop(self):
        scores_n = defaultdict(list)
        mean_n, std_n, p5_n, p95_n = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
        str_cols = [str(x) for x in list(range(self.config['min_db'], self.config['max_db'] + 1))] 
        for num_raters in self.m:
            print(f'NUMBER OF RATERS: {num_raters}', flush=True)

            for id_sample in range(self.config['S']):
                # Sampling raters (use id sample to use a different seed for each sample)
                raters_db = self.db_mos.loc[:, str_cols]
                raters_db = raters_db.sample(num_raters, axis=1, random_state=id_sample)
                #np.random.seed(id_sample)
                #sub_cols = np.random.choice(str_cols, num_raters)
                #raters_db = self.db_mos[sub_cols]
                
                # Concate subsample of raters
                db_mos = pd.concat([self.db_mos.drop([str(x) for x in list(range(self.config['min_db'], self.config['max_db'] + 1))], axis=1), raters_db], axis=1)
                
                # Calculate new MOS with rater subsample
                db_mos['MOS'] = raters_db.mean(axis=1)
                
                # Compute metrics
                scores = self.compute_metrics(db_mos)

                # Store for each sample size (S times)
                scores_n[num_raters].append(scores)
            
            sample_mean, sample_std, sample_p5, sample_p95 = self.metric_stats(scores_n[num_raters])
            mean_n[num_raters] = sample_mean
            std_n[num_raters] = sample_std
            p5_n[num_raters] = sample_p5
            p95_n[num_raters] = sample_p95

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