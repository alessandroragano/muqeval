import pandas  as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import tukey_hsd
import scipy.stats as stats
import numpy as np
import pingouin as pg
from utils import ci_mos_per_file

# Set style
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.set(font_scale = 2.1)

path = 'data/speech_mos_metrics.csv'
df = pd.read_csv(path)
cols = [str(x) for x in list(range(1, 25))]
#subset = cols + ['MOS', 'Condition', 'Filename', 'db']
df = df[df['db'] == 'TCD-VOIP']
df_ci = df.set_index('Filename')[cols]
df_ci.index.name = None
ci = ci_mos_per_file(df_ci)
df_melt = pd.melt(df, id_vars=['Filename'], value_vars=cols)
#res = pg.pairwise_tukey(data=df_melt, dv='value', between='Filename')
pass

# conf int (0,0) -1.71265, 0.2959784
