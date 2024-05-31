import scipy.stats as stats
import numpy as np
from collections import defaultdict
import pandas as pd

def relative_root_mean_squared_error(true, pred):
    num = np.sum(np.square(true - pred))
    den = np.sum(np.square(pred))
    squared_error = num/den
    rrmse_score = np.sqrt(squared_error)
    return rrmse_score
    
def metric_stats(metric_scores):
    ms_mean = np.mean(metric_scores)
    ms_std = np.std(metric_scores)
    ms_p5 = np.percentile(metric_scores, 5)
    ms_p95 = np.percentile(metric_scores, 95)
    return ms_mean, ms_std, ms_p5, ms_p95

def resampling(row, l=24):
    acr_values = row.value_counts().sort_index().index.to_list()
    p = row.value_counts().sort_index().values/l
    new_scores = np.random.choice(acr_values, size=l, p=p)
    new_scores = pd.Series(new_scores, index = list(map(str, range(1, l+1))))
    return new_scores

def ci_mos_per_file(listener_scores, C=0.95):
    """ Calculate the confidence interval 95% of mos scores using standard deviation (i.e. we do not consider the population but only the current sample)
    
    Parameters:
    mos (DataFrame): pandas dataframe of two columns (Filename, sample MOS)
    listener_scores (DataFrame): pandas dataframe where the first column is Filename and 
                                the rest is an MxN matrix with N equal to the number of votes per each sample
    C (float): desired confidence level
    df (int): degree of freedom
    
    Returns:
    ci (DataFrame): pandas dataframe with ci values for each sample
    """
    
    # Reset index
    listener_scores = listener_scores.reset_index().drop('index', axis=1)

    # Number of votes per each sample
    K = listener_scores.count(axis=1) 
    
    # Degree of freedom
    dof = K-1
    alpha = (1 - C)/2
    
    # MOS
    mos = listener_scores.mean(axis=1)

    # Create standard deviation array and t-distribution look up and cis
    sigma = np.zeros_like(mos)
    t = np.zeros_like(mos)
    ci_length = np.zeros_like(mos)
    ci_start = np.zeros_like(mos)
    ci_end = np.zeros_like(mos)
    for l, mos_l in enumerate(mos):
        # Calculate standard deviation
        sigma[l] = np.sqrt((np.sum((listener_scores.iloc[l,:]-mos_l)**2))/(K[l]-1)) # equivalent to np.std(listener_scores.iloc[l,:].to_numpy(), ddof=1)
        
        # Calculate confidence interval
        t[l] = stats.t.ppf(1-alpha, df=dof[l])
        ci_bar = t[l]*sigma[l]/np.sqrt(K[l])
        ci_length[l] = 2*ci_bar
        ci_start[l] = mos_l - ci_bar
        ci_end[l] = mos_l + ci_bar
    
    df_ci = {'Filename': listener_scores.index, 'ci_length': ci_length, 'ci_start': ci_start, 'ci_end': ci_end}
    df_ci = pd.DataFrame(df_ci)
    return df_ci

def ci_mos_per_cond(df, C=0.95):
    alpha = (1 - C)/2

    # Extract scores only
    listener_scores = df.drop(['Condition'], axis=1)
    listener_scores_with_cond = df
    
    # Number of votes per each sample
    K = listener_scores.count(axis=1).rename('K')

    # Initialize
    sigma, t, ci_length, ci_start, ci_end = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)

    # Group by conditions
    df_cond_group = listener_scores_with_cond.groupby('Condition')
    for cond_name, cond in df_cond_group:
        
        # Initialize std per file within same condition
        num = 0
        
        # MOS
        cond.drop('Condition', axis=1, inplace=True)
        mos = cond.mean(axis=1).rename('MOS')

        # Get numerator for each file and add
        for l, mos_l in enumerate(mos):
            num += np.sum((cond.iloc[l,:]-mos_l)**2)

        # Get total number of votes per condition
        N = K.loc[cond.index].sum()

        # Calculate std per condition
        sigma[cond_name] = np.sqrt(num/(N-1))

        # Calculate confidence interval per condition
        t[cond_name] = stats.t.ppf(1-alpha, df=N-1)
        ci_bar = t[cond_name]*sigma[cond_name]/np.sqrt(N)
        ci_length[cond_name] = 2*ci_bar
        ci_start[cond_name] = mos_l - ci_bar
        ci_end[cond_name] = mos_l + ci_bar
    
    df_ci = pd.DataFrame([ci_length, ci_start, ci_end]).T
    df_ci.rename({0: 'ci_length', 1: 'ci_start', 2: 'ci_end'}, axis=1, inplace=True)
    df_ci.index.name = 'Filename'
    return df_ci

# Test
# import os
# import pandas as pd
# root = '/media/alergn/hdd/datasets/audio/speech/TCD-VOIP/'
# lqs_file = os.path.join(root, 'subjective_scores.csv')
# df = pd.read_csv(lqs_file)
# df.set_index('Filename', inplace=True)
# #df_listener_scores = df.iloc[:,3:]
# df = pd.concat((df['ConditionID'], df.iloc[:,3:]), axis=1)
# df['Speaker'] = [i.split('.')[0][-2:] for i in df.index]
# df['DegCond'] = [i.split('_')[2] + '_' + str(df.loc[i,'ConditionID']) for i in df.index]

# ci = ci_mos_per_cond(df)
# print(ci)