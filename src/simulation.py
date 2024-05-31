import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
import random
sns.set(font_scale=2.2)
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
SEED = 0
np.random.seed(SEED)
random.seed(SEED)

xx = np.array([1, 4.8])
yy = np.array([1, 4.8])
means = [xx.mean(), yy.mean()]  
stds = [xx.std() / 3, yy.std() / 3]
corr = 0.8         
covs = [[stds[0]**2          , stds[0]*stds[1]*corr], 
        [stds[0]*stds[1]*corr,           stds[1]**2]] 

m = np.random.multivariate_normal(means, covs, 1000).T
fig = plt.figure(figsize=(10, 10))
sns.scatterplot(x=m[0], y=m[1])
plt.savefig(f'figs/simulation_{corr}.png')

# Plot deviation from population for each sample
pcc, srcc, ktau = [], [], []
num_drawns = 100
for sample in np.linspace(5, 300, 100, dtype=np.int32):
    for n in range(num_drawns):
        index = np.random.choice(list(range(0,1000)), sample)
        subset = m[:,index]
        pcc.append(pearsonr(subset[0], subset[1])[0])
        srcc.append(spearmanr(subset[0], subset[1])[0])
        ktau.append(kendalltau(subset[0], subset[1])[0])

# Plot each coefficient
fig = plt.figure(figsize=(10, 10))
x_ticks = np.repeat(np.linspace(5, 300, 100, dtype=np.int32), repeats=num_drawns)
ax = sns.scatterplot(x = x_ticks, y = pcc, s=200, edgecolor='black', alpha=0.8)
pop_pearson = np.round(pearsonr(m[0,:], m[1,:])[0], 3)
ax.axhline(y=pop_pearson, color='green', linewidth=2, label='PCC Population')
plt.ylabel('PCC')
plt.xlabel('Sample Size')
plt.legend()
plt.tight_layout()
plt.savefig(f'figs/pearsondev_{corr}.png')
plt.close()

fig = plt.figure(figsize=(10, 10))
ax = sns.scatterplot(x = x_ticks, y = srcc, s=200, edgecolor='black', alpha=0.8)
pop_spearman = np.round(spearmanr(m[0,:], m[1,:])[0], 3)
ax.axhline(y=pop_spearman, color='green', linewidth=2, label='SRCC Population')
plt.ylabel('SRCC')
plt.xlabel('Sample Size')
plt.legend()
plt.tight_layout()
plt.savefig(f'figs/spearmandev_{corr}.png')
plt.close()

fig = plt.figure(figsize=(10, 10))
ax = sns.scatterplot(x = x_ticks, y = ktau, s=200, edgecolor='black', alpha=0.8)
pop_ktau = np.round(kendalltau(m[0,:], m[1,:])[0], 3)
ax.axhline(y=pop_ktau, color='green', linewidth=2, label='KTAU Population')
plt.ylabel('KTAU')
plt.xlabel('Sample Size')
plt.legend()
plt.tight_layout()
plt.savefig(f'figs/ktaudev_{corr}.png')
plt.close()

# Intervals
import pandas as pd
intervals = sorted(list(set(pd.qcut(m[0], q=3).categories.left.to_list()) | set(pd.qcut(m[0], q=3).categories.right.to_list())))

for id, val in enumerate(intervals[1:]):
    subindex =  np.where((m[0] < val) & (m[0] > intervals[id]))[0]
    subdata = m[:,subindex]
    print(f'Interval: [{intervals[id]},{val}]')
    sub_pcc = pearsonr(subdata[0], subdata[1])[0]
    print(f'Pearson: {sub_pcc}')
    sub_srcc = spearmanr(subdata[0], subdata[1])[0]
    print(f'Spearman: {sub_srcc}')
    sub_ktau = kendalltau(subdata[0], subdata[1])[0]
    print(f'KTAU: {sub_ktau}\n')


fig = plt.figure(figsize=(12, 10))
sns.scatterplot(x=m[0], y=m[1], s=200, edgecolor='black')
plt.axvline(x=intervals[1], color='green', linewidth=4)
plt.axvline(x=intervals[2], color='green', linewidth=4)
#plt.axvline(x=intervals[3], color='green')
#plt.axvline(x=intervals[4], color='green')
plt.tight_layout()
plt.savefig(f'figs/simulation_vline_{corr}.png', dpi=250)