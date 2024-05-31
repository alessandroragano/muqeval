import pandas as pd
import os

stats = ['mean.csv', 'p5_n.csv', 'p95_n.csv', 'std.csv']
path = './experiments/exp1'

mean_p23exp1 = []
mean_p23exp3 = []
mean_tcdvoip = []

p5_p23exp1 = []

p95_df = []
std_df = []
db = []
model = []

for root, dir, files in os.walk(path):
    for f in files:
        if f in stats:
            model_db = root.split('/')[-1].split('_')
            model.append(model_db[0])
            if len(model_db) == 3:
                db = '_'.join(model_db[1:])
            else:
                db = model_db[1]
            
            path_file = pd.read_csv(os.path.join(root, f))
            if f == 'mean.csv':
                mean_p23exp1.append(path_file)
            elif f == 'p5_n.csv':
                p5_p23exp1.append(path_file)
            elif f == 'p95_n.csv':
                p95_df.append(path_file)
            elif f == 'std.csv':
                std_df.append(path_file)
            
mean_p23exp1 = pd.concat(mean_p23exp1).groupby('metric').mean().reset_index()
#mean_df.columns = sorted(mean_df.columns.values)
mean_p23exp1.to_csv(os.path.join(path, 'mean.csv'), index=False)
p5_p23exp1 = pd.concat(p5_p23exp1).groupby('metric').mean().reset_index()
#p5_df.columns = sorted(p5_df.columns.values)
p5_p23exp1.to_csv(os.path.join(path, 'p5_n.csv'), index=False)
p95_df = pd.concat(p95_df).groupby('metric').mean().reset_index()
#p95_df.columns = sorted(p95_df.columns.values)
p95_df.to_csv(os.path.join(path, 'p95_n.csv'), index=False)
std_df = pd.concat(std_df).groupby('metric').mean().reset_index()
#std_df.columns = sorted(std_df.columns.values)
std_df.to_csv(os.path.join(path, 'std.csv'), index=False)
pass