import pandas as pd
import os

stats = ['mean.csv', 'p5_n.csv', 'p95_n.csv', 'std.csv']
path = './experiments/exp1'

mean_df = []
p5_df = []
p95_df = []
std_df = []

for root, dir, files in os.walk(path):
    for f in files:
        if f in stats:
            path_file = pd.read_csv(os.path.join(root, f))
            if 'exp1' in path:
                mapping = dict(zip(path_file.drop(['metric', 'population'], axis=1).columns.to_list(), list(range(1,21))))
                path_file.rename(mapping, axis=1, inplace=True)
            
            if f == 'mean.csv':
                mean_df.append(path_file)
            elif f == 'p5_n.csv':
                p5_df.append(path_file)
            elif f == 'p95_n.csv':
                p95_df.append(path_file)
            elif f == 'std.csv':
                std_df.append(path_file)

mean_df = pd.concat(mean_df).groupby('metric').mean().reset_index()
#mean_df.columns = sorted(mean_df.columns.values)
mean_df.to_csv(os.path.join(path, 'mean.csv'), index=False)
p5_df = pd.concat(p5_df).groupby('metric').mean().reset_index()
#p5_df.columns = sorted(p5_df.columns.values)
p5_df.to_csv(os.path.join(path, 'p5_n.csv'), index=False)
p95_df = pd.concat(p95_df).groupby('metric').mean().reset_index()
#p95_df.columns = sorted(p95_df.columns.values)
p95_df.to_csv(os.path.join(path, 'p95_n.csv'), index=False)
std_df = pd.concat(std_df).groupby('metric').mean().reset_index()
#std_df.columns = sorted(std_df.columns.values)
std_df.to_csv(os.path.join(path, 'std.csv'), index=False)
pass