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
        
        # Number of splits
        self.n_splits = self.config['n_splits']
        if self.n_splits == 4:
            self.labels = ['Bad', 'Poor', 'Good', 'Excellent']
        elif self.n_splits == 2:
            self.labels = ['Bad', 'Excellent']

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
    

    def samples_loop(self):
        scores_n = defaultdict(list)
        rr_coefficient = defaultdict(list) # Range restricted coefficient
        str_cols = [str(x) for x in list(range(1, 25))] 
        self.db_mos['Range'] = pd.qcut(self.db_mos['MOS'], self.n_splits, labels=self.labels)
        db_range_grouped = self.db_mos.groupby('Range')

        for db_rg_name, db_mos in db_range_grouped:    
            if (db_rg_name == 'Bad') | (db_rg_name == 'Excellent'):           
                scores = self.compute_metrics(db_mos)

                #scores_n[db_rg_name].append(scores)
            
                rr_coefficient[db_rg_name] = scores
            # sample_mean, sample_std, sample_p5, sample_p95 = self.metric_stats(scores_n[num_raters])
            # mean_n[db_rg_name] = sample_mean
            # std_n[db_rg_name] = sample_std
            # p5_n[db_rg_name] = sample_p5
            # p95_n[db_rg_name] = sample_p95

        # Save stats
        rr_coefficient = pd.concat([pd.DataFrame(rr_coefficient['Bad']), pd.DataFrame(rr_coefficient['Excellent'])], axis=1)
        rr_coefficient.columns = ['Bad', 'Excellent']
        self.save_stats_csv(rr_coefficient, 'rr')

        # Save config file used for this experiment
        out_path_config = os.path.join(self.config['out_dir'], 'config.yaml') 
        with open(out_path_config, 'w') as out_config_file:
            _ = yaml.dump(self.config, out_config_file)
        pass    

    def save_stats_csv(self, df_stats, stats_name):
        stats_cat = pd.concat([df_stats, self.pop_df], axis=1)

        metric = self.config['metric']
        db = self.config['db']
        sub_dir = f'{metric}_{db}'
        out_path = os.path.join(self.config['out_dir'], sub_dir)
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        out_path = os.path.join(out_path, f'{stats_name}_{self.n_splits}.csv')
        stats_cat.reset_index().to_csv(out_path, index=False)