__author__ = 'lbernardi'

import matplotlib.pyplot as plt

from math import  *
import sklearn as sk
from sklearn import linear_model
import pandas as pd
from sklearn.dummy import DummyRegressor
import numpy as np

def ssCV(estimator, X, y , l = 10, m = 100, delta = 0.81):
    tps = int(ceil(float(m)/delta + 1))
    k = int(ceil(tps/(tps-m)))
    q = int(ceil(X.shape[0]/tps))
    preds = np.zeros((X.shape[0], l))


    E = np.array_split(range(X.shape[0]), X.shape[0]/tps)
    print 'q', q, 'len(E)', len(E), X.shape[0], X.shape[0]/tps
    print 'tps', tps, 'm', m
    print 'k', k

    for p in range(l):
        for i in range(q):
            np.random.shuffle(E[i])
            F = np.array_split(E[i], k)
            for j in range(k):
                S = E[i][np.in1d(E[i], F[j], invert=True)]
                training_set = S[np.random.choice(S.shape[0], m)]
                estimator.fit(X[training_set], y[training_set])
                print estimator.score(X[F[j]], y[F[j]])
                preds_Fj = estimator.predict(X[F[j]])
                for a, Xindex in enumerate(F[j]):
                    preds[Xindex, p] = preds_Fj[a]

                if len(E)>q and i==0 and j ==1:
                    print E[q+1]
                    preds_Eq_plus_1 = estimator.predict(X[E[q+1]])
                    for a, Xindex in enumerate(E[q+1]):
                        preds[Xindex, p] = preds_Eq_plus_1[a]

    return preds

#data = pd.read_csv('glass.csv', index_col='Id')
#print data



data = pd.read_csv('hour.csv', index_col='instant')
data = data.sample(frac=1).reset_index(drop=True)

features = ['mnth',  'hr', 'holiday',  'weekday',  'workingday', 'weathersit',   'atemp',   'hum',  'windspeed']
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
#estimator = make_pipeline(StandardScaler(), PolynomialFeatures(2), linear_model.LinearRegression(normalize=True,fit_intercept=True))
estimator = make_pipeline(StandardScaler(), PolynomialFeatures(1, interaction_only=not False, include_bias=not True), linear_model.SGDRegressor(fit_intercept=True, alpha = 0.4))
#estimator = make_pipeline(StandardScaler(), PolynomialFeatures(2, interaction_only=not False, include_bias=not True), linear_model.SGDRegressor(fit_intercept=True, alpha = 3))


#estimator = DummyRegressor(strategy='constant', constant=1)
#estimator = linear_model.LinearRegression(normalize=True,fit_intercept=True)
X = pd.get_dummies(data[features], columns=[a for a in ['weathersit', 'season', 'yr', 'mnth', 'hr', 'weekday'] if a in features],  drop_first=True)
y = data['cnt'].values
#y = np.log()
'''

X['holiday*weathersit']=X['weathersit_2']*X['weekday']
X['holiday*weathersit']=X['weathersit_3']*X['weekday']
X['holiday*weathersit']=X['weathersit_4']*X['weekday']

X['holiday*weathersit']=X['weathersit_2']*X['holiday']
X['holiday*weathersit']=X['weathersit_3']*X['holiday']
X['holiday*weathersit']=X['weathersit_4']*X['holiday']




X['windspeed*windspeed']=X['windspeed']*X['windspeed']
X['temp*temp']=X['atemp']*X['atemp']
X['hum*hum']=X['hum']*X['hum']


X['temp*windspeed']=X['atemp']*X['windspeed']
X['temp*hum']=X['atemp']*X['hum']
X['windspeed*hum']=X['windspeed']*X['hum']

X['temp*windspeed*hum']=X['atemp']*X['windspeed']*X['hum']
'''


ts = int(X.shape[0]*0.8)
R = ssCV(estimator, X.as_matrix(), y, m=ts)
print data.ix[np.argmax(np.var(R, axis =1))]
print sqrt(np.max(np.var(R, axis =1)))


#R = np.exp(R)
print R.shape
for fi in range(1):
    plt.scatter(data[features][features[fi]], y, color='red')
    for i in range(R.shape[1]):
        plt.scatter(data[features][features[fi]], R[:,i])



variance = np.mean(np.var(R, axis =1))
error = np.mean(np.mean((R.T - y).T**2, axis = 1))
bias_plus_noise = error - variance

print 'sqrd Variance error share', variance/error
print 'sqrd Variance', sqrt(variance)
print 'sqrd Bias+Noise', sqrt(np.mean(bias_plus_noise))
print 'RMSE', sqrt(error)

print '0000000'
from sklearn.metrics import mean_squared_error


estimator.fit(X[ts:], y[ts:])
print sqrt(mean_squared_error(estimator.predict(X[ts:]), y[ts:]))