import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import os

from loadData import loadData

files = os.listdir('MPIIMobileAttention/')
df = loadData(files[0:1])
print(files[0:1])

#features.head()


df = df.drop(columns=['trust_screen', 'environment', 'indoor_outdoor','trust_activity',
                       'user_name', 'message', 'app_description','stat_mobile',
                       'screenstatus', 'screenactivity', 'phone_screenonoff_description',
                       'sem_seg','subject_folder','block_folder', 'object_seg', 'question_type'])

#'evironment_1', 'evironment_2', 'evironment_3', 'evironment_4', 'evironment_5', 

#df = df.fillna(0)

#df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(axis='columns')

df = pd.get_dummies(df)

print(df.head())


# print("df: \n", df)
# print(df.dtypes)

# print (df)
X = df.loc[:, df.columns != 'gaze_on_screen']
X= np.array(X).astype(np.float32)
print(X.dtype)


y = np.array(df['gaze_on_screen'])


print('X: \n', X.dtype, X)
print('y: \n',y.dtype, y)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

from proximityMatrix import proximityMatrix

proxMat = proximityMatrix(clf, X_train, normalize=True)

print("proxMat: \n", proxMat)


