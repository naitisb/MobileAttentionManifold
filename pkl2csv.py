import pandas as pd
import numpy as np
import os

files = os.listdir('MPIIMobileAttention/')


for file in files:
    df = pd.read_pickle(os.path.join('MPIIMobileAttention/', file))
    newfile = file.replace('.pkl', '.csv')
    df.to_csv(os.path.join('MobileAttn/', newfile))
