#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 00:30:55 2017

@author: anasbarakat
"""
#BEST SCORE: 88.5906040268

#tous les clf à plus de 85%

import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

clf1 = QuadraticDiscriminantAnalysis()
clf1.fit(Xsuper_vrai,ysuper_vrai)
ypred1 = clf1.predict(Xtest_fu_aug_scFu)

clf2 = clfMLPfus
clf2.fit(Xsuper_vrai,ysuper_vrai)
ypred2 = clf2.predict(Xtest_fu_aug_scFu)

#BEST score max_features ='log2' and default   87.2483221477
clf3 = RandomForestClassifier(max_features ='log2') #BEST SCORE
clf3.fit(Xsuper_vrai,ysuper_vrai)
ypred3 = clf3.predict(Xtest_fu_aug_scFu)

#####Ajout#######
clf4 = ExtraTreesClassifier(max_features='log2')
clf4.fit(Xsuper_vrai,ysuper_vrai)
ypred4 = clf4.predict(Xtest_fu_aug_scFu)
########

def uniqueMultiY(y1,y2,y3):
    y_pred = np.zeros((n_test_files,1))
    
    t = features.shape[0]//Fus
    for i in range(298): 
        yconcat = np.concatenate((y1[i*t:(i+1)*t],y2[i*t:(i+1)*t]), axis = 0)
        yconcat = np.concatenate((yconcat,y3[i*t:(i+1)*t]), axis = 0)
        yi, counts = np.unique(yconcat, return_counts = True)
        y_pred[i] = yi[np.argmax(counts)]
    return(y_pred)
    
def uniqueMultiY2(y1,y2):
    y_pred = np.zeros((n_test_files,1))
    
    t = features.shape[0]//Fus
    for i in range(298): 
        yconcat = np.concatenate((y1[i*t:(i+1)*t],y2[i*t:(i+1)*t]), axis = 0)
        yi, counts = np.unique(yconcat, return_counts = True)
        y_pred[i] = yi[np.argmax(counts)]
    return(y_pred)

def uniqueMultiY4(y1,y2,y3,y4):
    y_pred = np.zeros((n_test_files,1))
    
    t = features.shape[0]//Fus
    for i in range(298): 
        yconcat = np.concatenate((y1[i*t:(i+1)*t],y2[i*t:(i+1)*t]), axis = 0)
        yconcat = np.concatenate((yconcat,y3[i*t:(i+1)*t]), axis = 0)
        yconcat = np.concatenate((yconcat,y4[i*t:(i+1)*t]), axis = 0)
        yi, counts = np.unique(yconcat, return_counts = True)
        y_pred[i] = yi[np.argmax(counts)]
    return(y_pred)
    
ypredMajVote = uniqueMultiY(ypred1,ypred2,ypred3)
np.savetxt('ypredMajVote.txt', ypredMajVote, fmt='%d')


#feature 50 (0.006672)
#features 55,58,48,59,39,53,52,32,36,54,37,34,33,35
toDelete = [55,58,48,59,39,53,52,32,36,54,37,34,33,35]
Xtrain = Xsuper_vrai
Xtrain = np.delete(Xtrain,toDelete,axis=1)

Xtest = Xtest_fu_aug_scFu
Xtest = np.delete(Xtest,toDelete,axis=1)

    

#toDelete3 = [53,52,32,36,54,37,34,33,35] #88.5906040268
toDelete3 = [57,47,30,31,38,50]
toDelete3 = np.concatenate((toDelete,toDelete3))
Xtrain3 = np.delete(Xtrain,toDelete3,axis=1)
Xtest3 = np.delete(Xtest,toDelete3,axis=1)

### BEST SCORE 89.2617449664 avec feature selection with Random Forest using 
#Xtrain and Xtest
############ feature selection- feature importance RF ################
#importances = forest.feature_importances_
#std = np.std([tree.feature_importances_ for tree in forest.estimators_],
#             axis=0)
#indices = np.argsort(importances)[::-1]

# Print the feature ranking
#print("Feature ranking:")

#for f in range(60):
#    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
#plt.figure()
#plt.title("Feature importances")
#plt.bar(range(60), importances[indices],
#       color="r", yerr=std[indices], align="center")
#plt.xticks(range(60), indices)
#plt.xlim([-1, 60])
#plt.show()


##### score 88.5906040268 MLP avec Xtrain et Xtest (ie feature selection)
