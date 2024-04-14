from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd 
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

#Loads dataset
iris = load_iris()
data = iris.data
label = iris.target

testing = {'accuracy', 'f1_weighted', 'roc_auc_ovr'}

kf = KFold(n_splits=5, random_state=None, shuffle=True)

#SVC
svc_model = SVC(probability=True)

print("SVC")

for s in testing:
    scores = cross_val_score(svc_model, data, label, scoring=s, cv=kf)
    print(s + ": "+ str(np.mean(scores)))

print("--------------------")

#Naieve Bayes
nb_model = GaussianNB()

print("Naieve Bayes")
for s in testing:
    scores = cross_val_score(nb_model, data, label, scoring=s, cv=kf)
    print(s + ": "+ str(np.mean(scores)))
print("--------------------")

#Random Forest
rf_model = RandomForestClassifier()

print("Random Forest")
for s in testing:
    scores = cross_val_score(rf_model, data, label, scoring=s, cv=kf)
    print(s + ": "+ str(np.mean(scores)))
print("--------------------")

#XGBoost
xg_model = xgb.XGBClassifier()

print("XGBoost")
for s in testing:
    scores = cross_val_score(xg_model, data, label, scoring=s, cv=kf)
    print(s + ": "+ str(np.mean(scores)))
          
print("--------------------")

#K-nearest Neighbors
knn_model = KNeighborsClassifier()

print("K-nearest Neighbors")
for s in testing:
    scores = cross_val_score(knn_model, data, label, scoring=s, cv=kf)
    print(s + ": "+ str(np.mean(scores)))
          
print("--------------------")
