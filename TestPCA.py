__author__ = 'lbernardi'

import numpy as np
import sklearn as sk
import pandas as pd
from sklearn import linear_model
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt

from scipy import linalg
from sklearn.datasets import load_iris
from math import *
from sklearn.decomposition import PCA
from sklearn.utils.extmath import svd_flip
from scipy.stats import zscore






ds = pd.read_csv('hour.csv', index_col='instant')

features = ['yr',  'mnth',  'hr', 'holiday',  'weekday',  'atemp',   'hum',  'windspeed']

X = ds[features]
X = pd.get_dummies(X, columns=['yr', 'mnth', 'hr', 'weekday'], drop_first=True)
X.apply(zscore)
y = ds['cnt']

X = ds[features].as_matrix()
k = X.shape[1]

mean = np.mean(X, axis = 0)

X -= mean


n = X.shape[0]

#corrmat = np.corrcoef(X, rowvar=0)
#C = np.dot(X.T, X)/11
U, S, V = linalg.svd(X, full_matrices=False)
U, V = svd_flip(U, V, u_based_decision=True)

U = U[:,:k]
V = V[:,:k]
S = S[:k]


principal_components = U*S
principal_directions = V
eigenvectors = V.T
eigenvalues = S**2/(n-1)
loadings = eigenvectors*np.sqrt(eigenvalues)

#loadings = eigenvectors
print loadings[:,:2]*1000

explained_variance_ = (S ** 2) / (n-1)
total_var = explained_variance_.sum()
explained_variance_ratio_ = explained_variance_ / total_var

print str(explained_variance_ratio_)

corrmat =  np.corrcoef(X, rowvar=0)
for i, feature_name_i in enumerate(features):
    for j, feature_name_j in enumerate(features):
        if corrmat[i,j] > 0.7 and i !=j:
            print feature_name_i, feature_name_j, corrmat[i,j]



fig, ax = plt.subplots()
ax.scatter(loadings[:,0],loadings[:,1])

for i, txt in enumerate(features):
    ax.annotate(features[i], (loadings[i,0], loadings[i,1]))
plt.show()


loadings, T = fr.rotate_factors(loadings, 'quartimax')
fig, ax = plt.subplots()
ax.scatter(loadings[:,0],loadings[:,1])

for i, txt in enumerate(features):
    ax.annotate(features[i], (loadings[i,0], loadings[i,1]))
plt.show()

'''
print sorted(explained_variance_)

#print np.dot(np.dot(V.T, np.diag(S)/11), V)


'''