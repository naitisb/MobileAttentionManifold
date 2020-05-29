import pandas as pd
import numpy as np
import os

def loadData(files):

    list_of_dfs = []

    for file in files:
        df = pd.read_pickle(os.path.join('MPIIMobileAttention/', file))
        list_of_dfs.append(df)

    big_df = pd.concat(list_of_dfs, ignore_index = True, sort=True)

    return big_df

# df = loadData(os.listdir('MPIIMobileAttention/'))



