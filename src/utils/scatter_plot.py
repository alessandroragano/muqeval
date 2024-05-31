import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import sys
from utils import ci_mos_per_file, ci_mos_per_cond
from scipy.spatial import distance_matrix
import numpy as np
sns.set(font_scale=1.6)

def cci_preprocessing(mos_p, listener_scores, digits, tau, per_cond, plot, fig_path):
    # Get CI
    if per_cond:
        df_ci = ci_mos_per_cond(listener_scores)
        mos = listener_scores.reset_index().set_index(['Filename', 'DegCond']).mean(axis=1).groupby(level=['DegCond']).mean().rename('MOS')
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
        y_values = masked_pred_diff/(masked_gt_diff + np.finfo(np.float).eps)
        x_values = gt_dist[mask]
        pair_name = ['Concordant' if y_val > 0 else 'Discordant' for y_val in y_values] 
        df_plot = pd.DataFrame({'y': y_values, 'x': x_values, 'Pair Name': pair_name})
        g = sns.scatterplot(data=df_plot, x='x', y='y', hue='Pair Name', s=120, legend=True)
        plt.rcParams['text.usetex'] = True
        g.get_legend().set_title(None)
        plt.xlabel('|$y_i$-$y_ j$|')
        plt.ylabel(r'slope: $\frac{\hat{y_i} - \hat{y_j}}{y_i-y_j}$')
        plt.tight_layout()
        plt.xlim([0,4.0])
        plt.ylim([-4.0, 4.0])
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

metric = 'PESQ'
db = 'PESQ_23_EXP1'

sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
df = pd.read_csv('data/speech_mos_metrics.csv')
#df = df[df['db'] == db]

# ALL 
pcc_all, _ = pearsonr(df['MOS'], df[metric])
print(f'PCC all: {pcc_all}')
sns.scatterplot(data=df, x='MOS', y=metric, s=50, edgecolor='black')
plt.xlim([1, 5])
plt.ylim([1, 5])
plt.tight_layout()
plt.savefig(f'figs/{metric}_scatter.png', dpi=200)
plt.close()

df_scatter = pd.melt(df, id_vars=['Filename', 'db', 'MOS'], value_vars=['PESQ', 'VISQOL'], var_name=['Metrics'], value_name='Pred MOS')
df_scatter['Model DB'] =  df_scatter[['Metrics', 'db']].agg('_'.join, axis=1)
plt.figure(figsize=(9, 9))
g = sns.scatterplot(data=df_scatter, x='MOS', y='Pred MOS', hue='Model DB', style='Model DB', markers=['o', '^', 'v', 'D', 's', 'X'], s=120)
for lh in g.legend_.legendHandles: 
    lh.set_alpha(1)
    lh._sizes = [150] 
plt.tight_layout()
plt.savefig(f'figs/all_scatter.png', dpi=200)

# 1 - 2
df1 = df[(df['MOS'] > 1.0) & (df['MOS'] <=2.0)]
pcc_1, _ = pearsonr(df1['MOS'], df1[metric])
print(f'PCC 1-2 : {pcc_1}')

# 4 - 5
df4 = df[(df['MOS'] >= 4.0) & (df['MOS'] <=5.0)]
pcc_4, _ = pearsonr(df4['MOS'], df4[metric])
print(f'PCC 4-5 : {pcc_4}')

# CCI all
rater_list = [str(x) for x in list(range(0,100))]
rater_columns = [val for val in df.columns if val in rater_list]
raters = df[rater_columns]
raters.columns = raters.columns.astype(int)

cci_all = cci(df[metric], raters, digits=3)
print(f'CCI all: {cci_all}')

# CCI 1-2
raters1 = raters.loc[df1.index]
cci_1 = cci(df1[metric], raters1, digits=3)
print(f'CCI 1-2: {cci_1}')

# CCI 4-5
raters4 = raters.loc[df4.index]
cci_4 = cci(df4[metric], raters4, digits=3)
print(f'CCI 4-5: {cci_4}')
