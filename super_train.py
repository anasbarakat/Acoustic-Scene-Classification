#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 01:06:57 2017

@author: MacBilal
"""

# Best score 87.2483221477
# New best score 87.9194630872  

import os
import librosa
import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import preprocessing

qda = QuadraticDiscriminantAnalysis()

# max_iter = 100 donne le même score  

qda.fit(X_con_aug_scFu,y_con)
y_pconqda=qda.predict(Xtest_fu_aug_scFu)
y_rconqda=uniqueFus(y_pconqda)

# np.savetxt('y_rconMLPfus.txt', y_rconMLPfus, fmt='%d')

L = []
for i in range(298):
    if y_rconqda[i] == y_rconMLPfus[i]:
        L = L + [i]


FILEROOT = "./audio/"
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
    if i in L:
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
    
            Xvrai = features_fu.T
        else:
            for j in range(features.shape[0]//Fus):
                if j==0:
                    features_fu = np.mean(features[Fus*j:Fus*(j+1)],0)
                else: 
                    features_fu = np.c_[features_fu,np.mean(features[Fus*j:Fus*(j+1)],0)]
            #print(mfcc_fu.shape)
        
            Xvrai = np.r_[Xvrai, features_fu.T]
            


# Xvrai.shape[0]/float(240)

yvrai = []


for i in range(298):
    if y_rconqda[i] == y_rconMLPfus[i]:
        yvrai=np.append(yvrai,y_rconqda[i])


y_v = []

for i in range(len(yvrai)):
    temp = yvrai[i]*np.array([1]*(938//Fus))
    y_v=np.concatenate((y_v,temp),axis=0)


ysuper_vrai = np.r_[y_con,y_v]
Xsuper_vrai = np.r_[X_con_augFu,Xvrai]
Xsuper_vrai = preprocessing.scale(Xsuper_vrai)


# BEST CLASSIFIER
#clfMLPfus = MLPClassifier(solver='lbfgs',alpha=5e-8, hidden_layer_sizes=(200), random_state=1,
#                     max_iter= 100, early_stopping = True,
#                      verbose = True)

clfMLPfus = MLPClassifier(solver='lbfgs',alpha=1e-8, hidden_layer_sizes=(200), random_state=1,
                       max_iter= 100, early_stopping = True,
                       verbose = True)

# max_iter = 100 donne le même score  

clfMLPfus.fit(Xsuper_vrai,ysuper_vrai)
y_pconMLPfus=clfMLPfus.predict(Xtest_fu_aug_scFu)
y_rconMLPfus=uniqueFus(y_pconMLPfus)
#np.savetxt('ysuper_vrai.txt', y_rconMLPfus, fmt='%d')


# qda 86.5771812081

#error = []
#conf = confusion_matrix(y_con,y_confusion)
#for i in range(15):
#    error += [np.sum(conf[i,:])-conf[i,i]]
    
#random forest : 85.2348993289

# majority vote sur 3 classifieus score 86.5771812081
#clf1 = QuadraticDiscriminantAnalysis()
#clf2 = clfMLPfus
#clf3 = RandomForestClassifier()
#eclf1 = VotingClassifier(estimators=
#[('qda', clf1), ('MLP', clf2), ('rf', clf3)], voting='hard')
#eclf1.fit(Xsuper_vrai,ysuper_vrai)
#eclf1.predict(Xtest_fu_aug_scFu)
#ypredMajVot = eclf1.predict(Xtest_fu_aug_scFu)
#ypredMajVotFin = uniqueFus(ypredMajVot)
#np.savetxt('yMajVot.txt', ypredMajVotFin, fmt='%d')
