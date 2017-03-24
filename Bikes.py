__author__ = 'lbernardi'
import statsmodels.formula.api as smf
import numpy as np
import pandas as  pd
from scipy.stats import zscore
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import statsmodels.api as sm
def corrmat(X):
    #return np.eye(X.shape[1])
    return np.corrcoef(X, rowvar=0)
    Xmat = X.as_matrix()
    A = np.zeros((Xmat.shape[1], Xmat.shape[1]))
    for i in range(Xmat.shape[1]):
        for j in range(Xmat.shape[1]):
            A[i,j]=spearmanr(Xmat[:,i], Xmat[:,j])[0]
    return A

def plot_correlation(X):
    global figi
    col_names = X.columns.values
    A = np.abs(corrmat(X[col_names]))

    cluser_ass = AffinityPropagation(affinity='precomputed').fit_predict(A)
    print sorted(zip(cluser_ass, col_names))
    col_names = np.array(col_names)[np.argsort(cluser_ass)]
    A = np.abs(corrmat(X[col_names]))

    fig, ax = plt.subplots()
    ax.matshow(A[:, :])
    plt.xticks(range(len(col_names[:])), col_names[:], rotation='90', fontsize=12)
    plt.yticks(range(len(col_names[:])), col_names[:], fontsize=12)
    plt.tight_layout()

    plt.show()
    #plt.savefig('mc_%s.png' % (figi))
    figi+=1

figi = 0

data = pd.read_csv('hour.csv', index_col='instant')
'''

features = ['season', 'yr',  'mnth',  'hr', 'holiday',  'weekday',  'workingday', 'weathersit',  'temp',   'atemp',   'hum',  'windspeed']



#First try, we get about 9 inconclusive factors
ds = data
X = ds[features]
X = pd.get_dummies(X, columns=['weathersit', 'season', 'yr', 'mnth', 'hr', 'weekday'])
X  = X.rename(columns={'season_1':'winter', 'season_2':'spring', 'season_3':'summer', 'season_4':'fall'})
X  = X.rename(columns={'weathersit_1':'sunny', 'weathersit_2':'cloudy', 'weathersit_3':'snowy', 'weathersit_4':'rainy'})
X  = X.rename(columns={'mnth_1': 'jan', 'mnth_2':'feb', 'mnth_3':'mar', 'mnth_4':'apr', 'mnth_5':'may', 'mnth_6':'jun', 'mnth_7':'jul', 'mnth_8':'aug', 'mnth_9':'sep', 'mnth_10':'oct', 'mnth_11':'nov', 'mnth_12':'dec'})
X  = X.rename(columns={'yr_0':'2011', 'yr_1':'2012'})


y = ds['cnt']
print smf.OLS(y, X).fit().summary()
plot_correlation(X)

#After fixing the dummy variables we still get 5 non significant factors
ds = data
X = ds[features]
X = pd.get_dummies(X, columns=['weathersit', 'season', 'yr', 'mnth', 'hr', 'weekday'], drop_first=True)
X  = X.rename(columns={'season_1':'winter', 'season_2':'spring', 'season_3':'summer', 'season_4':'fall'})
X  = X.rename(columns={'weathersit_1':'sunny', 'weathersit_2':'cloudy', 'weathersit_3':'snowy', 'weathersit_4':'rainy'})
X  = X.rename(columns={'mnth_1': 'jan', 'mnth_2':'feb', 'mnth_3':'mar', 'mnth_4':'apr', 'mnth_5':'may', 'mnth_6':'jun', 'mnth_7':'jul', 'mnth_8':'aug', 'mnth_9':'sep', 'mnth_10':'oct', 'mnth_11':'nov', 'mnth_12':'dec'})
X  = X.rename(columns={'yr_0':'2011', 'yr_1':'2012'})

y = ds['cnt']
print smf.OLS(y, X).fit().summary()
plot_correlation(X)

#remove temp
ds = data
features = ['season', 'yr',  'mnth',  'hr', 'holiday',  'weekday',  'workingday', 'weathersit',  'atemp',   'hum',  'windspeed']

X = ds[features]
X = pd.get_dummies(X, columns=['weathersit', 'season', 'yr', 'mnth', 'hr', 'weekday'], drop_first=True)
X  = X.rename(columns={'season_1':'winter', 'season_2':'spring', 'season_3':'summer', 'season_4':'fall'})
X  = X.rename(columns={'weathersit_1':'sunny', 'weathersit_2':'cloudy', 'weathersit_3':'snowy', 'weathersit_4':'rainy'})
X  = X.rename(columns={'mnth_1': 'jan', 'mnth_2':'feb', 'mnth_3':'mar', 'mnth_4':'apr', 'mnth_5':'may', 'mnth_6':'jun', 'mnth_7':'jul', 'mnth_8':'aug', 'mnth_9':'sep', 'mnth_10':'oct', 'mnth_11':'nov', 'mnth_12':'dec'})
X  = X.rename(columns={'yr_0':'2011', 'yr_1':'2012'})

y = ds['cnt']
print smf.OLS(y, X).fit().summary()
plot_correlation(X)

#remove season
ds = data
features = ['yr',  'mnth',  'hr', 'holiday',  'weekday',  'workingday', 'weathersit',  'atemp',   'hum',  'windspeed']

X = ds[features]
X = pd.get_dummies(X, columns=['weathersit', 'yr', 'mnth', 'hr', 'weekday'], drop_first=True)
X  = X.rename(columns={'season_1':'winter', 'season_2':'spring', 'season_3':'summer', 'season_4':'fall'})
X  = X.rename(columns={'weathersit_1':'sunny', 'weathersit_2':'cloudy', 'weathersit_3':'snowy', 'weathersit_4':'rainy'})
X  = X.rename(columns={'mnth_1': 'jan', 'mnth_2':'feb', 'mnth_3':'mar', 'mnth_4':'apr', 'mnth_5':'may', 'mnth_6':'jun', 'mnth_7':'jul', 'mnth_8':'aug', 'mnth_9':'sep', 'mnth_10':'oct', 'mnth_11':'nov', 'mnth_12':'dec'})
X  = X.rename(columns={'yr_0':'2011', 'yr_1':'2012'})

y = ds['cnt']
print smf.OLS(y, X).fit().summary()
plot_correlation(X)


#remove weathersit
ds = data.sample(frac = 1)
features = ['yr',  'mnth',  'hr', 'holiday',  'weekday',  'workingday',  'atemp',   'hum',  'windspeed']
X = ds[features]
X = pd.get_dummies(X, columns=['yr', 'mnth', 'hr', 'weekday'], drop_first=True)
X  = X.rename(columns={'season_1':'winter', 'season_2':'spring', 'season_3':'summer', 'season_4':'fall'})
X  = X.rename(columns={'weathersit_1':'sunny', 'weathersit_2':'cloudy', 'weathersit_3':'snowy', 'weathersit_4':'rainy'})
X  = X.rename(columns={'mnth_1': 'jan', 'mnth_2':'feb', 'mnth_3':'mar', 'mnth_4':'apr', 'mnth_5':'may', 'mnth_6':'jun', 'mnth_7':'jul', 'mnth_8':'aug', 'mnth_9':'sep', 'mnth_10':'oct', 'mnth_11':'nov', 'mnth_12':'dec'})
X  = X.rename(columns={'yr_0':'2011', 'yr_1':'2012'})

y = ds['cnt']
print smf.OLS(y, X).fit().summary()
plot_correlation(X)

#remove workingday
ds = data.sample(frac = 1)
features = ['yr',  'mnth',  'hr', 'holiday',  'weekday',  'atemp',   'hum',  'windspeed']

X = ds[features]
X = pd.get_dummies(X, columns=['yr', 'mnth', 'hr', 'weekday'], drop_first=True)
X  = X.rename(columns={'season_1':'winter', 'season_2':'spring', 'season_3':'summer', 'season_4':'fall'})
X  = X.rename(columns={'weathersit_1':'sunny', 'weathersit_2':'cloudy', 'weathersit_3':'snowy', 'weathersit_4':'rainy'})
X  = X.rename(columns={'mnth_1': 'jan', 'mnth_2':'feb', 'mnth_3':'mar', 'mnth_4':'apr', 'mnth_5':'may', 'mnth_6':'jun', 'mnth_7':'jul', 'mnth_8':'aug', 'mnth_9':'sep', 'mnth_10':'oct', 'mnth_11':'nov', 'mnth_12':'dec'})
X  = X.rename(columns={'yr_0':'2011', 'yr_1':'2012'})

y = ds['cnt']
print smf.OLS(y, X).fit().summary()
plot_correlation(X)

'''

ds = data.sample(frac = 1)
features = ['yr',  'mnth',  'hr', 'holiday',  'weekday',  'atemp',   'hum',  'windspeed']

X = ds[features]
X = pd.get_dummies(X, columns=['yr', 'mnth', 'hr', 'weekday'], drop_first=True)
X  = X.rename(columns={'season_1':'winter', 'season_2':'spring', 'season_3':'summer', 'season_4':'fall'})
X  = X.rename(columns={'weathersit_1':'sunny', 'weathersit_2':'cloudy', 'weathersit_3':'snowy', 'weathersit_4':'rainy'})
X  = X.rename(columns={'mnth_1': 'jan', 'mnth_2':'feb', 'mnth_3':'mar', 'mnth_4':'apr', 'mnth_5':'may', 'mnth_6':'jun', 'mnth_7':'jul', 'mnth_8':'aug', 'mnth_9':'sep', 'mnth_10':'oct', 'mnth_11':'nov', 'mnth_12':'dec'})
X  = X.rename(columns={'yr_0':'2011', 'yr_1':'2012'})

y = np.array(ds['cnt'].values)
X = X.as_matrix()
reg =  smf.OLS(y, X).fit()

e = reg.resid**2

w = 1./smf.OLS(e, reg.fittedvalues).fit().fittedvalues


wX = (X.T*w).T

wy = y*w

wls = smf.OLS(wy, wX).fit()
print wls.summary()
plt.scatter(wy, wls.fittedvalues)
plt.show()