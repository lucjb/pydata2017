__author__ = 'lbernardi'
import statsmodels.formula.api as smf
import numpy as np
import pandas as  pd
from scipy.stats import zscore
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
def corrmat(X):
    return np.corrcoef(X, rowvar=0)
    Xmat = X.as_matrix()
    A = np.zeros((Xmat.shape[1], Xmat.shape[1]))
    for i in range(Xmat.shape[1]):
        for j in range(Xmat.shape[1]):
            A[i,j]=spearmanr(Xmat[:,i], Xmat[:,j])[0]
    return A


def pr(x, y):
    mx = x.mean()
    my = y.mean()
    xm, ym = x-mx, y-my
    r_num = np.add.reduce(xm * ym)
    r_den = np.sqrt(np.sum(xm*xm) * np.sum(ym*ym))
    r = r_num / r_den
    return r

data = pd.read_csv('hotel_ufi_country.tsv', sep='\t', nrows=10000)

def plot_correlation(X):
    col_names = X.columns.values
    A = np.abs(corrmat(X[col_names]))
    print A.shape
    cluser_ass = AffinityPropagation(affinity='precomputed').fit_predict(A)
    print sorted(zip(cluser_ass, col_names))
    col_names = np.array(col_names)[np.argsort(cluser_ass)]
    A = np.abs(corrmat(X[col_names]))

    fig, ax = plt.subplots()
    ax.matshow(A[:, :],  aspect=1)
    #plt.xticks(range(len(col_names[:])), col_names[:], rotation='90', fontsize=18)
    #plt.yticks(range(len(col_names[:])), col_names[:], fontsize=18)

    plt.tight_layout()
    plt.show()

    #plt.savefig('mc_%s.png' % (figi))

X = pd.get_dummies(data, columns = ['city', 'country'], drop_first=True)

plot_correlation(X)

plt.matshow(np.abs(np.corrcoef(X, rowvar=0)), interpolation='none')
plt.show()