from zipfile import ZipFile
from urllib.request import urlopen
import pandas as pd
import os

def download_url(url, save_path):
    with urlopen(url) as dl_file:
        with open(save_path, 'wb') as out_file:
            out_file.write(dl_file.read())

URL = 'https://datasets.d2.mpi-inf.mpg.de/MobileHCI2018/MPIIMobileAttention.zip'

download_url(URL, '../MPIIMobileAttention.zip')

print ('url downloaded :) \n')

# open and save the zip file onto computer
url = urlopen(URL)
output = open('./MPIIMobileAttention.zip', 'wb')    # note the flag:  "wb"  
print ('opened \n')      
output.write(url.read())
print ('written \n')  
output.close()
print ('closed \n')  

# read the zip file as a pandas dataframe
df = pd.read_pickle('MPIIMobileAttention.zip')    # zip files       
print ('read a pkl \n')  

df.to_pickle("./MPIIMobileAttention.pkl")
print ('made a pkl \n')  
# if keeping on disk the zip file is not wanted, then:
# os.remove(zipName)   # remove the copy of the zipfile on disk