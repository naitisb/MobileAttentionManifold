from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os, sys, getopt, pdb
from numpy import *
from numpy.linalg import *
from numpy.random import *
import pylab

namelist = ['Data_P1_RB1.pkl', 'Data_P1_RB2.pkl', 'Data_P1_RB3.pkl']
list_of_dfs = []

for name in namelist:
    df = pd.read_pickle('MPIIMobileAttention/' + name)
    list_of_dfs.append(df)

big_df = pd.concat(list_of_dfs)

big_df.to_pickle("JointPreProcessedPkls/Data_P1.pkl")

df = big_df.replace([np.inf, -np.inf], np.nan)

"""
df_num = df.select_dtypes(include=['number'])

df_bool = df.select_dtypes(include=['bool'])

df = pd.concat([df_num.mask(df_num >= 3E38, np.nan), df_bool], axis = 1)
"""

df = df.select_dtypes(include=['number', 'bool'])

df = df.mask(df >= 3E38, np.nan)

df = df.dropna(axis=1, how='all')

df.loc[:, df.ne(0).any()]

#df = df.dropna(axis=0)

df.fillna(df.mean())

phone_df = df.loc[:,(df.columns.str.startswith("phone_")) | (df.columns.str.startswith("app_")) |
                    (df.columns.str.startswith("touch_")) | (df.columns.str.startswith("gps_")) |
                    (df.columns.str.startswith("screen")) | (df.columns.str.startswith("disp")) |
                    (df.columns.str.startswith("whatsapp")) | (df.columns.str.startswith("temp")) |
                    (df.columns.str.startswith("distance_cam"))]
rgb_df1 = df.loc[:,(df.columns.str.startswith("objectclass_")) | (df.columns.str.startswith("objectness_")) |
                    (df.columns.str.startswith("saliency_")) | (df.columns.str.startswith("segmentationclass_"))]
rgb_df2 = df[['gaze_xundist', 'gaze_yundist', 'corner1_xundist', 'corner1_yundist',
            'corner2_xundist', 'corner2_yundist', 'corner3_xundist', 'corner3_yundist',  'corner4_xundist',
             'corner4_yundist', 'corner1_xundistext', 'corner1_yundistext', 'corner2_xundistext',
             'corner2_yundistext','corner3_xundistext', 'corner3_yundistext', 'corner4_xundistext',
             'corner4_yundistext','corner1ext_x', 'corner1ext_y', 'corner2ext_x', 'corner2ext_y',
             'corner3ext_x', 'corner3ext_y','corner4ext_x', 'corner4ext_y', 'face_detections_world']]
rgb_df = pd.concat([rgb_df1, rgb_df2], axis=1)

#'object_seg', 'sem_seg',

headimu_df = df.loc[:,(df.columns.str.startswith("accelerometer_")) | (df.columns.str.startswith("gyro_")) |
                    (df.columns.str.startswith("mobilephone_in_scene_vid")) | (df.columns.str.startswith("corner1_x")) |
                    (df.columns.str.startswith("corner1_y")) | (df.columns.str.startswith("corner2_y")) |
                    (df.columns.str.startswith("corner2_x")) | (df.columns.str.startswith("corner3_x")) |
                    (df.columns.str.startswith("corner3_y")) | (df.columns.str.startswith("corner4_x")) |
                    (df.columns.str.startswith("corner4_y")) ]

depth_df = df.loc[:,df.columns.str.startswith("depth_")]

gaze_df = df[['depth', 'pupil_x', 'pupil_y','gaze_x', 'gaze_y', 'diameter', 'major', 'minor', 'angle']]
# ,'fix_dispersions', 'fix_durations', 'fix_centroids_y', 'fix_centroids_x',
# 'fix_centroidsext_x', 'fix_centroidsext_y,''saliency', 'objectness',
egocentric_df = pd.concat([headimu_df, pd.concat([rgb_df, depth_df], axis = 1)], axis = 1) # the egocentric sensors if we want to revisit as the paper did

X = pd.concat([headimu_df, pd.concat([rgb_df, pd.concat([depth_df, pd.concat([phone_df, gaze_df], axis = 1)], axis = 1)], axis = 1)], axis = 1)
# X = pd.get_dummies(X)
X= np.array(X).astype(np.float32)
# X = X[X < 1E308]

y = np.array(df['gaze_on_screen'])

print('X: \n', X.dtype, X)
print('y: \n',y.dtype, y)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

print('X_train: \n', X_train.dtype, X_train)
print('X_test: \n', X_test.dtype, X_test)

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=75)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

# METRICS
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score

probs = clf.predict_proba(X_test)
gaze_on_probs = probs[:,1]
fpr, tpr, thresholds = roc_curve(y_test, gaze_on_probs)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'y', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

y_pred = clf.predict(X_test)
pre_score = precision_score(y_test, y_pred)
re_score= recall_score(y_test, y_pred)
f1_score = f1_score(y_test, y_pred)
print('F1 score: ' + f1_score)

"""
precision, recall, threshold = precision_recall_curve(y_test, y_pred)
average_precision = average_precision_score(y_test, y_pred)
step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
plt.step(recall, precision, color='r', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2, color='r', **step_kwargs)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
"""

from proximityMatrix import proximityMatrix

proxMat = proximityMatrix(clf, X_train, normalize=True)

print("proxMat: \n", proxMat)

from mds import mds, square_points

Y, eigs = mds(proxMat)

pylab.figure(1)
pylab.plot(Y[:,0],Y[:,1],'.')

pylab.figure(2)
pylab.plot(points[:,0], points[:,1], '.')

pylab.show()
