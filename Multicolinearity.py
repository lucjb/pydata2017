__author__ = 'lbernardi'

import numpy as np
import sklearn as sk
import pandas as pd
from sklearn import linear_model
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt

import factor_rotation as fr

from math import *

data = pd.read_csv('hour.csv', index_col='instant')



features = ['season', 'yr',  'mnth',  'hr', 'holiday',  'weekday',  'workingday', 'weathersit',  'temp',   'atemp',   'hum',  'windspeed']
#feaures = ['mnth',  'hr',  'temp',   'windspeed']

scores = []
coefs = []
for  _ in range(10):
    ds = data.sample(frac = 0.5)
    X = ds[features]
    y = ds['cnt']

    model = linear_model.LinearRegression(normalize=True,fit_intercept=True)
    #model = linear_model.Ridge(normalize=True, fit_intercept=True, alpha = 0.1, solver = 'cholesky')
    #model = linear_model.BayesianRidge(normalize=True,fit_intercept=True)
    #model = linear_model.ARDRegression(normalize=True,fit_intercept=True)
    #model = linear_model.HuberRegressor()
    model.fit(X, y)
    scores.append(model.score(X, y))
    coefs.append(model.coef_)

from sklearn.decomposition import PCA


pca = PCA(n_components=2)
X = data[features]
X = X - np.mean(X, axis=0)
pca.fit(X)
print pca.explained_variance_ratio_, sum(pca.explained_variance_ratio_)

V = pca.components_.T
S = np.diag(np.sqrt(X.shape[0]*pca.explained_variance_))

loadings = np.dot(V, S)/sqrt(X.shape[0])
print loadings


L, T = fr.rotate_factors(loadings.T,'varimax')
print L.T


'''
plt.scatter(pca.transform(data[features])[:,0], pca.transform(data[features])[:,1])
plt.show()



res = data['cnt'].values - model.predict(data[features])
for f in features:
    plt.scatter(data[f].values, res)
    plt.show()

scores = np.array(scores)
coefs = np.array(coefs).reshape((10, len(features)))
means = coefs.mean(axis=0)
stds = coefs.std(axis=0)

for i in range(len(features)):
    print str(means[i]) + ' +/- ' + str(stds[i])

print str(scores.mean()) + ' +/- ' + str(scores.std())


corrmat = np.corrcoef(data[features], rowvar=0)
selected = set(features)
blacklisted = set()
for i in range(len(features)):
    for j in range(len(features)):
        corr = corrmat[i][j]
        if corr>0.75:
            corrmat[i][j]=1
        else:
            corrmat[i][j]=0

A = np.corrcoef(data[features], rowvar=0)
cluser_ass = AffinityPropagation(affinity='precomputed', damping = 0.5).fit_predict(A)
print zip(np.array(features)[np.argsort(cluser_ass)], np.sort(cluser_ass))
'''
'''
0.5
20.0224031595 +/- 1.52031516629
80.9124107778 +/- 1.66147581944
-0.112289780543 +/- 0.42145366621
7.66867891598 +/- 0.132004734804
-20.9547941336 +/- 3.44532284731
1.88037129052 +/- 0.604047670942
3.79883757334 +/- 2.7072622006
-4.04049832641 +/- 2.13742165822
77.367867118 +/- 42.3423994003
237.156682146 +/- 44.837940553
-196.382741499 +/- 9.70432303966
39.3033023242 +/- 9.68341362335
0.389783551096 +/- 0.00730264422051

15.7917791679 +/- 0.53932575896
74.7294265979 +/- 2.05181863783
1.02128905403 +/- 0.321962455373
7.05022088291 +/- 0.107108181115
-21.1347959834 +/- 5.64322098197
1.83250529811 +/- 0.276348917729
3.939154601 +/- 1.65052716434
-5.51628837434 +/- 1.35832248318
131.052944398 +/- 4.54945212319
164.246116083 +/- 4.47396572716
-185.935705693 +/- 4.2063623876
41.2243617471 +/- 6.32023397478
0.38808436484 +/- 0.0050529396424

'''
