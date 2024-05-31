import pandas as pd

df = pd.read_csv('/media/alergn/hdd/github/cimos/data/image_mos_metrics.csv', encoding='latin-1', sep=';')
df.rename({'Filepath_Deg': 'Filepath Deg', 
           'refFile': 'Filepath Ref', 
           'x1': '1', 'x2': '2', 'x3': '3', 'x4': '4', 'x5': '5', 'x6': '6', 'x7': '7', 'x8': '8', 'x9': '9', 'x10': '10', 'x11': '11', 'x12': '12', 'x13': '13', 'x14': '14', 'x15': '15', 'x16': '16'
           }, axis=1, inplace=True)
df.drop(['psnr', 'ssimProcess', 'psnrProcess', 'brisqueProcess'], axis=1, inplace=True)
df.to_csv('/media/alergn/hdd/github/cimos/data/img_mos_metrics.csv', sep=',')
pass