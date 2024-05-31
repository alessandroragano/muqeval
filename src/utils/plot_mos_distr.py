import pandas  as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, anderson, kstest
import scipy.stats as stats
import numpy as np

# Set style
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.set(font_scale = 2.1)

path = 'data/speech_mos_metrics.csv'
df = pd.read_csv(path)
cols = [str(x) for x in list(range(1, 25))]
dbs = df.groupby('db')

for db_name, db in dbs:
    # PLOT DISTRIBUTIONS
    plt.figure(figsize=(10, 8))
    sns.histplot(data=db['MOS'], stat='probability', legend=False, kde=True)
    plt.xlabel('MOS')
    plt.ylabel('Probability')
    plt.tight_layout()
    plt.savefig(f'figs/{db_name}_distribution.pdf', dpi=150)
    plt.close()

    # PLOT ACR DISTRIBUTIONS
    plt.figure(figsize=(10, 8))
    db_melt = pd.melt(db[cols], value_vars=cols)
    sns.histplot(data=db_melt['value'], stat='count', legend=False, kde=False)
    plt.xlabel('ACR')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'figs/{db_name}_ACR_distribution.pdf', dpi=150)
    plt.close()    

    # Gaussianity tests
    print(f'MOS - DB: {db_name}')
    # Shapiro *** p_value < 0.05 Reject normality ***
    _, p_value_shapiro = shapiro(db['MOS'].values)
    if p_value_shapiro < 0.05:
        print(f'pvalue {np.round(p_value_shapiro, 5)} < 0.05 -> Reject Normality')
    else:
        print(f'pvalue {np.round(p_value_shapiro, 5)} >= 0.05 -> Accept Normality')
    
    # Anderson *** p_value > 0.05 Reject normality ***
    p_value_anderson = anderson(db['MOS'].values)[1][2]
    if p_value_anderson > 0.05:
        print(f'pvalue {np.round(p_value_anderson, 5)} > 0.05 -> Reject Normality')
    else:
        print(f'pvalue {np.round(p_value_anderson, 5)} <= 0.05 -> Accept Normality')    
    
    # Kolmogorov *** p_value < 0.05 Reject normality ***
    _, p_value_kstest = kstest(db['MOS'].values, stats.norm.cdf)
    if p_value_kstest < 0.05:
        print(f'pvalue {np.round(p_value_kstest, 5)} < 0.05 -> Reject Normality')
    else:
        print(f'pvalue {np.round(p_value_kstest, 5)} >= 0.05 -> Accept Normality')    

    # Gaussianity tests
    print(f'ACR - DB: {db_name}')
    # Shapiro *** p_value < 0.05 Reject normality ***
    _, p_value_shapiro = shapiro(db_melt['value'].values)
    if p_value_shapiro < 0.05:
        print(f'pvalue {np.round(p_value_shapiro, 5)} < 0.05 -> Reject Normality')
    else:
        print(f'pvalue {np.round(p_value_shapiro, 5)} >= 0.05 -> Accept Normality')
    
    # Anderson *** p_value > 0.05 Reject normality ***
    p_value_anderson = anderson(db_melt['value'].values)[1][2]
    if p_value_anderson > 0.05:
        print(f'pvalue {np.round(p_value_anderson, 5)} > 0.05 -> Reject Normality')
    else:
        print(f'pvalue {np.round(p_value_anderson, 5)} <= 0.05 -> Accept Normality')    
    
    # Kolmogorov *** p_value < 0.05 Reject normality ***
    _, p_value_kstest = kstest(db_melt['value'].values, stats.norm.cdf)
    if p_value_kstest < 0.05:
        print(f'pvalue {np.round(p_value_kstest, 5)} < 0.05 -> Reject Normality')
    else:
        print(f'pvalue {np.round(p_value_kstest, 5)} >= 0.05 -> Accept Normality')    