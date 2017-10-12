#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 00:02:20 2017

@author: anasbarakat
"""

# score Fus=9:   81.8791946309
# score Fus=5:   80.53
# score Fus=6: 79.1946308725
# score Fus=3: 75.8389261745
# score Fus=1: 73.1543624161
# score Fus=8: 80.5369127517
# score Fus=11: 83.8926174497 (validation score arrive à 0.90) BEST
# score Fus=12: 80.5369127517 (validation score atteint 0.916642)
# score Fus=13: 80.5369127517 (validation score atteint 0.905717)
# score Fus=14: 82.5503355705

import os
import librosa
import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier

from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

Fus = 11 #BEST  BEST SCORE 89,....

FILEROOT="./audio/"
annotations = pd.read_table(os.path.join(FILEROOT, "train.txt"),sep=r"\s+",
                            names=['file', 'label'])
#
## select data
## print annotations.loc[annotations.label=='beach']
#
#
labels = {'beach': 0, 'bus': 1, 
          'cafe/restaurant': 2,'car': 3,'city_center':4, 'forest_path':5,
          'grocery_store':6,'home':7,
          'library':8,'metro_station':9, 'office':10,'park':11, 
         'residential_area':12, 'train':13, 'tram':14}

for i, afile in annotations.iterrows():
       
    y, sr = librosa.load(os.path.join("./" ,afile.file), sr=None)
    mfcc = librosa.feature.mfcc(y=y, n_fft=512, hop_length=512, n_mfcc=20)
    #plt.figure(figsize=(10, 4))
#    #librosa.display.specshow(librosa.power_to_db(np.abs(mfcc), ref=np.max),
#    #                         y_axis='mel', x_axis='time')
    #print("mfcc shape", mfcc.shape)
    mfcc_delta = librosa.feature.delta(mfcc)
    #print("mfcc_delta shape",mfcc_delta.shape)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    features = np.r_[mfcc,mfcc_delta]
    features = np.r_[features,mfcc_delta2]
    features = features.T
    
    if i==0:
        for j in range(features.shape[0]//Fus):
            if j==0:
                features_fu = np.mean(features[Fus*j:Fus*(j+1)],0)
            else: 
                features_fu = np.c_[features_fu,np.mean(features[Fus*j:Fus*(j+1)],0)]
        #print(mfcc_fu.shape)

        X_fu_augFu = features_fu.T
        y_fu = labels[afile.label]*np.array([1]*(features.shape[0]//Fus))
    else:
        for j in range(features.shape[0]//Fus):
            if j==0:
                features_fu = np.mean(features[Fus*j:Fus*(j+1)],0)
            else: 
                features_fu = np.c_[features_fu,np.mean(features[Fus*j:Fus*(j+1)],0)]
        #print(mfcc_fu.shape)
    
        X_fu_augFu = np.r_[X_fu_augFu, features_fu.T]
        y_fu = np.concatenate((y_fu, 
                           labels[afile.label]*np.array([1]*(features.shape[0]//Fus))),
                           axis = 0) 

###########################################################################

FILEROOT="./audio/"
annotations = pd.read_table(os.path.join(FILEROOT, "dev.txt"),sep=r"\s+",
                            names=['file', 'label'])

# select data
## print annotations.loc[annotations.label=='beach']
#
#
labels = {'beach': 0, 'bus': 1, 
          'cafe/restaurant': 2,'car': 3,'city_center':4, 'forest_path':5,
          'grocery_store':6,'home':7,
          'library':8,'metro_station':9, 'office':10,'park':11, 
         'residential_area':12, 'train':13, 'tram':14}

for i, afile in annotations.iterrows():
       
    y, sr = librosa.load(os.path.join("./" ,afile.file), sr=None)
    mfcc = librosa.feature.mfcc(y=y, n_fft=512, hop_length=512, n_mfcc=20)
    #plt.figure(figsize=(10, 4))
#    #librosa.display.specshow(librosa.power_to_db(np.abs(mfcc), ref=np.max),
#    #                         y_axis='mel', x_axis='time')
    mfcc_delta = librosa.feature.delta(mfcc)
    
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    features = np.r_[mfcc,mfcc_delta]
    features = np.r_[features,mfcc_delta2]

    features = features.T
    
    if i==0:
        for j in range(features.shape[0]//Fus):
            if j==0:
                features_fu = np.mean(features[Fus*j:Fus*(j+1)],0)
            else: 
                features_fu = np.c_[features_fu,np.mean(features[Fus*j:Fus*(j+1)],0)]

        Xval_fu_augFu = features_fu.T
        yval_fu = labels[afile.label]*np.array([1]*(features.shape[0]//Fus))
        
    else:
        for j in range(features.shape[0]//Fus):
            if j==0:
                features_fu = np.mean(features[Fus*j:Fus*(j+1)],0)
            else: 
                features_fu = np.c_[features_fu,np.mean(features[Fus*j:Fus*(j+1)],0)]
        #print(mfcc_fu.shape)
    
        Xval_fu_augFu = np.r_[Xval_fu_augFu, features_fu.T]
        yval_fu = np.concatenate((yval_fu, 
                           labels[afile.label]*np.array([1]*(features.shape[0]//Fus))),
                           axis = 0) 
    
    
    
    
FILEROOT = './audio/'
annotations = pd.read_table(os.path.join(FILEROOT, "test_files.txt"),sep=r"\s+",
                            names=['file', 'label'])

# select data
# print annotations.loc[annotations.label=='beach']


labels = {'beach': 0, 'bus': 1, 
          'cafe/restaurant': 2,'car': 3,'city_center':4, 'forest_path':5,
          'grocery_store':6,'home':7,
          'library':8,'metro_station':9, 'office':10,'park':11, 
         'residential_area':12, 'train':13, 'tram':14}

for i, afile in annotations.iterrows():
       
    y, sr = librosa.load(os.path.join("./" ,afile.file), sr=None)
    mfcc = librosa.feature.mfcc(y=y, n_fft=512, hop_length=512, n_mfcc=20)
    #plt.figure(figsize=(10, 4))
    #librosa.display.specshow(librosa.power_to_db(np.abs(mfcc), ref=np.max),
    #                         y_axis='mel', x_axis='time')
    mfcc_delta = librosa.feature.delta(mfcc)
    
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    features = np.r_[mfcc,mfcc_delta]
    features = np.r_[features,mfcc_delta2]

    features = features.T
    
    if i==0:
        for j in range(features.shape[0]//Fus):
            if j==0:
                features_fu = np.mean(features[Fus*j:Fus*(j+1)],0)
                #print(mfcc_fu.shape)
            else: 
                features_fu = np.c_[features_fu,np.mean(features[Fus*j:Fus*(j+1)],0)]
        #print(mfcc_fu.shape)

        Xtest_fu_augFu = features_fu.T
    else:
        for j in range(features.shape[0]//Fus):
            if j==0:
                features_fu = np.mean(features[Fus*j:Fus*(j+1)],0)
            else: 
                features_fu = np.c_[features_fu,np.mean(features[Fus*j:Fus*(j+1)],0)]
        #print(mfcc_fu.shape)
    
        Xtest_fu_augFu = np.r_[Xtest_fu_augFu, features_fu.T]

#########################
##### fusion-concatenate ########

X_con_augFu=np.r_[X_fu_augFu,Xval_fu_augFu]
y_con = np.r_[y_fu,yval_fu]

#scaling
X_con_aug_scFu = preprocessing.scale(X_con_augFu)    
Xtest_fu_aug_scFu = preprocessing.scale(Xtest_fu_augFu)  

def uniqueFus(y):
    c = 0 
    y_u=np.empty(0)
    while(c<len(y)):
        li=[]
        for i in range(features.shape[0]//Fus):
            li = li+ [y[c+i]]
        y_u= np.append(y_u,max(set(li), key=li.count))

        c=c+features.shape[0]//Fus
    return y_u
   
#ls =200, adam 5e-8 max_iter=100 BEST SCORE
ls=200
clfMLPfus = MLPClassifier(solver='lbfgs',alpha=5e-8, hidden_layer_sizes=(ls), random_state=1,
                       max_iter= 100, early_stopping = True,
                       verbose = True)

#lstest = 200
#clfMLPfus = MLPClassifier(solver='lbfgs',alpha=5e-8, hidden_layer_sizes=(lstest,lstest,lstest), random_state=1,
#                       max_iter= 100, early_stopping = True,
#                       verbose = True)
#### score 82.5503355705 (3 couches de 512 neurones lbfgs)



# alpha=1e-7 best score 83.8926174497
# alpha=1e-8 same (best score 1)
# max_iter = 100 donne le même score que 10 (au début) 

# alpha = 5e-7: score 82.5503355705
# alpha = 5e-8: score 84.5637583893  BEST SCORE (en validation score atteint: 0.910146)

# alpha = 1e-5: score 83.2214765101

#idée d'optim: rajouter mellog at d'autres (energy ...)

#### GrisSearchCV ######
#parameters = {'alpha':[1e-9,5e-9,3e-8,5e-8,7e-8,1e-7,1e-6,1e-5]}
#parameters = {'alpha':[5e-6,1e-5,3e-5,5e-5,1e-4]}

#clfSVCgrid = GridSearchCV(clfMLPfus, parameters)
#clfSVCgrid.fit(X_con_aug_scFu,y_con)
#print(clfSVCgrid.best_params_)

clfMLPfus.fit(X_con_aug_scFu,y_con)
y_pconMLPfus=clfMLPfus.predict(Xtest_fu_aug_scFu)
y_rconMLPfus=uniqueFus(y_pconMLPfus)
#np.savetxt('y_rconMLPfus.txt', y_rconMLPfus, fmt='%d')