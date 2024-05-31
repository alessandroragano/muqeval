import numpy as np
import scipy.stats as stats
from src.utils.utils import ci_mos_per_file, ci_mos_per_cond
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import combinations
from scipy.spatial import distance_matrix
sns.set_style('darkgrid')

# This script reproduces statistical metrics reported in ITU-T P.1401
# 7.5.1 RMSE (accuracy of the model)
def rmse(mos, mos_p):
    # Number of samples
    N = mos.shape[0]

    # Absolute prediction error
    p_err = mos - mos_p

    # Root mean squared error
    rmse = np.sqrt((1/(N-1))*np.sum(p_err**2))
    return rmse

# 7.5.2.1 Outlier ratio (consistency of the model)
def out_ratio(mos_p, listener_scores, digits=2):
    # Absolute prediction error
    mos = listener_scores.mean(axis=1).round(digits)
    mos_p = mos_p.round(digits)
    p_err = mos - mos_p

    # Number of subjects per sample
    n_subj = len(listener_scores.columns)

    # Number of samples
    N = mos.shape[0]

    # Check variable a is Gaussian or t-distribution
    if n_subj>=30:
        # Gaussian
        a = 1.96
    else:
        # Look up t-distribution table
        C = 0.95
        alpha = (1-C)/2
        a = stats.t.ppf(alpha, n_subj-1)
    
    # Calculate how many outliers are present in the prediction error
    n_outliers = 0 
    for i in p_err:
        if abs(i) > a * np.std(mos)/np.sqrt(n_subj):
            n_outliers += 1
    
    # Calculate outlier ratio
    return n_outliers/N

# 7.5.3 Pearson correlation (linearity of the scores)
def pcc(mos, mos_p):
    pcc_stats, _ = stats.pearsonr(mos, mos_p)
    return np.round(pcc_stats, decimals=3)

# Spearman correlation (ranking of the scores)
def srcc(mos, mos_p):
    srcc_stats, _ = stats.spearmanr(mos, mos_p)
    return np.round(srcc_stats, decimals=3)

# Kendall's Tau  
def ktau(mos, mos_p):
    ktau_stats, _ = stats.kendalltau(mos, mos_p)
    return np.round(ktau_stats, decimals=3)

# 7.7 Epsilon-insensitive RMSE (takes into account the uncertainity of the listeners)
def eps_rmse(mos_p, listener_scores, ddof=1, digits=2):
    # Get confidence intervals
    ci_bar = ci_mos_per_file(listener_scores)

    # Calculate epsilon-insensitive absolute prediction error
    mos = listener_scores.mean(axis=1).round(digits)
    mos_p = mos_p.round(digits)
    p_err = np.max(0, np.abs(mos-mos_p)-2*ci_bar)

    # Calculate rmse star
    N = mos.shape[0]
    rmse_star = np.sqrt((1/(N-ddof))*np.sum(p_err**2))
    
    return rmse_star

def all_pairs(lst):
    if len(lst) <= 1:
        return []
     
    pairs = [(lst[0], x) for x in lst[1:]]
     
    return pairs + all_pairs(lst[1:])

def cci_preprocessing(mos_p, listener_scores, digits, tau, per_cond, plot, fig_path):
    # Get CI
    if per_cond:
        df_ci = ci_mos_per_cond(listener_scores)
        mos = listener_scores.set_index([listener_scores.index, 'Condition']).mean(axis=1).groupby(level=['Condition']).mean().rename('MOS')
    else:
        df_ci = ci_mos_per_file(listener_scores, C=0.90)
        # Get mos
        mos = listener_scores.mean(axis=1).round(digits)

    # Get threshold
    #if tau == None:
    #    tau = df_ci['ci_length'].mean().round(digits)

    # Get mask (i.e. indices corresponding to the pairs where distance is greater than tau)
    x = mos.values.reshape(-1, 1)
    y = mos_p.values.reshape(-1, 1)
    gt_dist = distance_matrix(x, x)
    # mask = gt_dist > tau

    # NEW (FIND NON OVERLAPPING CONFIDENCE INTERVALS)
    
    # Calculate pairwise matrix of difference (not distance as above)
    ci_dist_sign = (df_ci['ci_start'].values.reshape(-1,1)  - df_ci['ci_end'].values)
    
    # Extract upper and lower diagonal matrices
    upper = np.triu(ci_dist_sign)
    lower = np.tril(ci_dist_sign)
    
    # Mirror w.r.t the diagonal and sign inversion
    lower_mirrored = -1*np.rot90(np.fliplr(lower))
    
    # Find where signs are concordant i.e., where each pair CI does not overlap 
    concordant_signs = np.triu((np.sign(upper) == np.sign(lower_mirrored)))

    # Make it symmetric
    concordant_signs = 1*concordant_signs + 1*concordant_signs.T - np.diag(np.diag(1*concordant_signs))
    mask = np.array(concordant_signs, dtype=bool)

    # # Melt into a 3 column dataframe
    # df_csi = pd.DataFrame(concordant_signs)
    # df_csi = df_csi.where(np.triu(np.ones(df_csi.shape)).astype(np.bool_))
    # df_csi = df_csi.stack().reset_index()

    # Get gt and pred matrix differences (i.e. include sign)
    gt_diff = (x[:,None,:] - x[None,:,:]).sum(axis=-1)
    pred_diff = (y[:,None,:] - y[None,:,:]).sum(axis=-1)
    
    # Mask both predictions and ground truth
    masked_pred_diff = pred_diff[mask]
    masked_gt_diff = gt_diff[mask]

    # Calculate concordants. Transform to zero all the predictions that are equal (mos_i == mos_j) is considered as misranking
    concordants = (np.sign(masked_pred_diff)*np.sign(masked_gt_diff) + 1) / 2
    
    # When moslqo (i) = moslqo (j), np.sign returns zero so concordants = 0.5. These have to be considered misranking and so are set to 0.0
    concordants[concordants==0.5] = 0.0
    total = masked_gt_diff.shape[0]

    if plot:
        sns.set(font_scale=1.4)
        y_values = masked_pred_diff/(masked_gt_diff + np.finfo(float).eps)
        x_values = gt_dist[mask]
        pair_name = ['Concordant' if y_val > 0 else 'Discordant' for y_val in y_values] 
        df_plot = pd.DataFrame({'y': y_values, 'x': x_values, 'Pair Name': pair_name})
        g = sns.scatterplot(data=df_plot, x='x', y='y', hue='Pair Name', s=120, legend=True)
        plt.rcParams['text.usetex'] = True
        g.get_legend().set_title(None)
        plt.xlabel('|$y_i$-$y_j$|')
        plt.ylabel(r'slope: $\frac{\hat{y_i} - \hat{y_j}}{y_i-y_j}$')
        plt.tight_layout()
        plt.xlim([0, 4.0])
        plt.ylim([-7.0, 7.0])
        plt.savefig(fig_path)  
        plt.close()

    return concordants, total, mask, x, y

# Concordance Index MOS (FAST, NUMPY BROADCAST APPROACH)
def cci(mos_p, listener_scores, digits=2, plot=False, fig_path=None, tau=None, per_cond=False):
    
    concordants, total, _, _, _ =  cci_preprocessing(mos_p, listener_scores, digits, tau, per_cond, plot, fig_path)
    #print(f'CONCORDANTS: {concordants.sum()}')
    #print(f'TOTAL: {total}')

    # Get cci
    cci_score = concordants.sum() / total
    cci_score = np.round(cci_score, digits)
    
    return cci_score

def wcci (mos_p, listener_scores, digits=2, plot=False, fig_path=None, tau=None, per_cond=False):
    concordants, total, mask, x, y = cci_preprocessing(mos_p, listener_scores, digits, tau, per_cond, plot, fig_path)

    # Normalize ground truth and predictions between 0 and 1
    x_norm = (np.clip(x, a_min=1, a_max=5) -1)/4
    y_norm = (np.clip(y, a_min=1, a_max=5) -1)/4

    # Get distance matrices
    gt_dist_norm = distance_matrix(x_norm, x_norm)
    pred_dist_norm = distance_matrix(y_norm, y_norm)

    # Calculate weight based on the distances
    w = 1 - abs(gt_dist_norm - pred_dist_norm)

    # Mask weight matrix
    w = w[mask]

    # Compute weighted concordants
    w_concordants = w*concordants

    # Get wcci
    wcci_score = w_concordants.sum() / total
    wcci_score = np.round(wcci_score, digits)

    return wcci_score

# Pairwise Ranking Accuracy (SLOW, LOOP APPROACH)
# def cimos(mos_p, listener_scores, digits=2, plot=False, wcimos=False):
#     # Get confidence intervals
#     df_ci = ci_mos_per_file(listener_scores)

#     # Get threshold
#     tau = df_ci['ci_length'].mean().round(digits)
#     #print(f'TAU: {tau}')

#     # Get mos
#     mos = listener_scores.mean(axis=1).round(digits)

#     # cimos variables
#     num_concordant_pairs = 0
#     num_concordant_pairs_wcimos = 0
#     size_constrained_set = 0

#     # Plot variables
#     if plot:
#         slope = []
#         gt_dist_list = []
#         pair_name = []

#     # Weighted version of cimos
#     w_sum = 0

#     #for (idx1, mos1), (idx2, mos2) in zip(mos[:-1].items(), mos[1:].items()):
#     for comb in combinations(mos.index, 2):
#         # Get pair idx
#         idx1 = comb[0]
#         idx2 = comb[1]

#         # Get pair mos values
#         mos1 = mos.loc[idx1]
#         mos2 = mos.loc[idx2]
#         if abs(mos1 - mos2) > tau:
#             if (idx1 in mos_p.index) & (idx2 in mos_p.index):
#                 # Update size of S
#                 size_constrained_set += 1

#                 # Extract predictions of the pair
#                 mos_p1 = mos_p.loc[idx1]
#                 mos_p2 = mos_p.loc[idx2]

#                 # Get GT and pred distances
#                 gt_dist = mos1 - mos2
#                 pred_dist = mos_p1 - mos_p2
#                 if type(pred_dist) == pd.Series:
#                     pred_dist = pred_dist[0]

#                 # Calculate the numerator of cimos
#                 val = (np.sign(gt_dist)*np.sign(pred_dist) + 1)/2
#                 #val = np.sign(gt_dist)*np.sign(pred_dist)

#                 if (wcimos) & (val == 1):
#                     # Normalize distances (assuming that max distance is 4 and min distance is 0 in ACR scale)
#                     norm_mos1 = (np.clip(mos1, a_min=1, a_max=5) - 1)/(4)
#                     norm_mos2 = (np.clip(mos2, a_min=1, a_max=5) - 1)/(4)
#                     abs_gt_dist_norm = abs(norm_mos1 - norm_mos2)
#                     norm_mos_p1 = (np.clip(mos_p1, a_min=1, a_max=5) - 1)/(4)
#                     norm_mos_p2 = (np.clip(mos_p2, a_min=1, a_max=5) - 1)/(4)
#                     abs_pred_dist_norm = abs(norm_mos_p1 - norm_mos_p2)

#                     w = abs(abs_gt_dist_norm - abs_pred_dist_norm)
#                     if val == 1:
#                         w = 1 - w
#                     w_sum += w
#                     w_val = w*val
#                 else:
#                     w_val = 0
                
#                 num_concordant_pairs += val
#                 num_concordant_pairs_wcimos += w_val
#                 # Update plot variables
#                 if plot:
#                     slope.append(np.round((pred_dist)/(gt_dist), digits))
#                     gt_dist_list.append(abs(mos1-mos2))
#                     if val==0:
#                         pair_name.append('Discordant')
#                     else:
#                         pair_name.append('Concordant')
    
#     # Plot
#     if plot:
#         df = pd.DataFrame(list(zip(gt_dist_list, slope, pair_name)), columns=['x', 'y', 'Pair Name'])
#         fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
#         sns.scatterplot(data=df, x='x', y='y', hue='Pair Name', s=120, ax=axs[0], legend=False)  
#         g = sns.kdeplot(data=df, x='x', y='y', hue='Pair Name', fill=True, ax=axs[1], legend=True)
#         plt.rcParams['text.usetex'] = True
#         g.get_legend().set_title(None)
#         axs[0].set_xlabel('|$y_i$-$y_ j$|')
#         axs[1].set_xlabel('|$y_i$-$y_j$|')
#         axs[0].set_ylabel(r'slope: $\frac{\hat{y_i} - \hat{y_j}}{y_i-y_j}$')
#         axs[1].set_ylabel(r'slope: $\frac{\hat{y_i} - \hat{y_j}}{y_i-y_j}$')
#         plt.tight_layout()
#         plt.savefig('./figures/cimos.png')

#     #if wcimos:
#     #    size_constrained_set = w_sum
#     #size_constrained_set = 2/(size_constrained_set*(size_constrained_set - 1))
#     cimos = np.round(num_concordant_pairs/size_constrained_set, decimals=3)
#     wcimos = np.round(num_concordant_pairs_wcimos/size_constrained_set, decimals=3)
#     return cimos, wcimos





# # Test
# import os
# import pandas as pd
# root = '/media/alergn/hdd/datasets/audio/speech/TCD-VOIP/'
# lqs_file = os.path.join(root, 'subjective_scores.csv')
# df = pd.read_csv(lqs_file)
# df.set_index('Filename', inplace=True)
# df_listener_scores = df.iloc[:,3:]
# #df_listener_scores.drop(['7', '8', '9', '10', '11', '12', '13', '14', '1', '2'], axis=1, inplace=True)
# mos = df_listener_scores.mean(axis=1).round(decimals=2) 

# # Test VISQOL scores
# root = '/media/alergn/hdd/datasets/audio/speech/TCD-VOIP/'
# lqo_file = os.path.join(root, 'visqol_results.csv')
# mos_p = pd.read_csv(lqo_file)
# mos_p['Filename'] = [s.split('/')[-1] for s in mos_p['degraded']]
# mos_p.set_index('Filename', inplace=True)
# mos_p = mos_p[['moslqo']]

# # Get cimos
# cimos_score = cimos(mos_p, df_listener_scores, wcimos=True)
# print(cimos_score)

# # Get PCC
# pc = pcc(mos, mos_p)
# print(pc)

# Simulation scenario