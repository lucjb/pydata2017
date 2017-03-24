__author__ = 'lbernardi'
import statsmodels.formula.api as smf
import numpy as np
import pandas as  pd
from scipy.stats import zscore
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt
from scipy import linalg

data = pd.read_csv('communities.data', na_values='?')
data = data.dropna()
print data
features = ["population", "householdsize", "racepctblack", "racePctWhite", "racePctAsian", "racePctHisp", "agePct12t21", "agePct12t29", "agePct16t24", "agePct65up", "numbUrban", "pctUrban", "medIncome", "pctWWage", "pctWFarmSelf", "pctWInvInc", "pctWSocSec", "pctWPubAsst", "pctWRetire", "medFamInc", "perCapInc", "whitePerCap", "blackPerCap", "indianPerCap", "AsianPerCap", "OtherPerCap", "HispPerCap", "NumUnderPov", "PctPopUnderPov", "PctLess9thGrade", "PctNotHSGrad", "PctBSorMore", "PctUnemployed", "PctEmploy", "PctEmplManu", "PctEmplProfServ", "PctOccupManu", "PctOccupMgmtProf", "MalePctDivorce", "MalePctNevMarr", "FemalePctDiv", "TotalPctDiv", "PersPerFam", "PctFam2Par", "PctKids2Par", "PctYoungKids2Par", "PctTeen2Par", "PctWorkMomYoungKids", "PctWorkMom", "NumIlleg", "PctIlleg", "NumImmig", "PctImmigRecent", "PctImmigRec5", "PctImmigRec8", "PctImmigRec10", "PctRecentImmig", "PctRecImmig5", "PctRecImmig8", "PctRecImmig10", "PctSpeakEnglOnly", "PctNotSpeakEnglWell", "PctLargHouseFam", "PctLargHouseOccup", "PersPerOccupHous", "PersPerOwnOccHous", "PersPerRentOccHous", "PctPersOwnOccup", "PctPersDenseHous", "PctHousLess3BR", "MedNumBR", "HousVacant", "PctHousOccup", "PctHousOwnOcc", "PctVacantBoarded", "PctVacMore6Mos", "MedYrHousBuilt", "PctHousNoPhone", "PctWOFullPlumb", "OwnOccLowQuart", "OwnOccMedVal", "OwnOccHiQuart", "RentLowQ", "RentMedian", "RentHighQ", "MedRent", "MedRentPctHousInc", "MedOwnCostPctInc", "MedOwnCostPctIncNoMtg", "NumInShelters", "NumStreet", "PctForeignBorn", "PctBornSameState", "PctSameHouse85", "PctSameCity85", "PctSameState85", "LemasSwornFT", "LemasSwFTPerPop", "LemasSwFTFieldOps", "LemasSwFTFieldPerPop", "LemasTotalReq", "LemasTotReqPerPop", "PolicReqPerOffic", "PolicPerPop", "RacialMatchCommPol", "PctPolicWhite", "PctPolicBlack", "PctPolicHisp", "PctPolicAsian", "PctPolicMinor", "OfficAssgnDrugUnits", "NumKindsDrugsSeiz", "PolicAveOTWorked", "LandArea", "PopDens", "PctUsePubTrans", "PolicCars", "PolicOperBudg", "LemasPctPolicOnPatr", "LemasGangUnitDeploy", "LemasPctOfficDrugUn", "PolicBudgPerPop"]

for  _ in range(1):
    ds = data.sample(frac = 1)
    X = ds[features]
    X = pd.get_dummies(X, drop_first=True)
    X.apply(zscore)


    print smf.OLS(ds['ViolentCrimesPerPop'], X).fit().summary()

A = np.corrcoef(ds[features], rowvar=0)
cluser_ass = AffinityPropagation(affinity='precomputed').fit_predict(A)
features = np.array(features)[np.argsort(cluser_ass)]
print zip(features, np.sort(cluser_ass))

A = np.corrcoef(ds[features], rowvar=0)
plt.matshow(A)
plt.xticks(range(len(features)), features, rotation='vertical')
plt.yticks(range(len(features)), features)
plt.show()

n = len(ds)
U, S, V = linalg.svd(X, full_matrices=False)


principal_components = U*S
principal_directions = V
eigenvectors = V.T
eigenvalues = S**2/(n-1)
loadings = eigenvectors*np.sqrt(eigenvalues)
loadings = eigenvectors**2

explained_variance_ = (S ** 2) / (n-1)
total_var = explained_variance_.sum()
explained_variance_ratio_ = explained_variance_ / total_var


print smf.OLS(ds['ViolentCrimesPerPop'], principal_components[:,:2]).fit().summary()

print str(explained_variance_ratio_)
plt.matshow(loadings[:,:10])
plt.show()
fig, ax = plt.subplots()
ax.scatter(loadings[:,0],loadings[:,1])

for i, txt in enumerate(features):
    ax.annotate(features[i], (loadings[i,0], loadings[i,1]))
plt.show()

'''
==============================================================================
Dep. Variable:                    cnt   R-squared:                       0.709
Model:                            OLS   Adj. R-squared:                  0.709
Method:                 Least Squares   F-statistic:                     3020.
Date:                Thu, 23 Mar 2017   Prob (F-statistic):               0.00
Time:                        18:52:04   Log-Likelihood:            -1.1073e+05
No. Observations:               17379   AIC:                         2.215e+05
Df Residuals:                   17365   BIC:                         2.216e+05
Df Model:                          14
Covariance Type:            nonrobust
================================================================================
                   coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------
season          20.0869      1.815     11.065      0.000      16.529      23.645
yr              80.7041      2.160     37.360      0.000      76.470      84.938
mnth            -0.0847      0.566     -0.150      0.881      -1.194       1.025
hr               7.7320      0.165     46.966      0.000       7.409       8.055
holiday        -22.7947      6.682     -3.411      0.001     -35.893      -9.697
weekday          1.8728      0.539      3.472      0.001       0.815       2.930
workingday       3.8362      2.391      1.605      0.109      -0.850       8.522
temp            84.8099     36.894      2.299      0.022      12.494     157.125
atemp          226.4076     41.444      5.463      0.000     145.173     307.642
hum           -195.7700      6.855    -28.560      0.000    -209.206    -182.334
windspeed       47.2782      9.624      4.913      0.000      28.414      66.142
weathersit_1   -34.6195      7.152     -4.841      0.000     -48.637     -20.602
weathersit_2   -23.3501      7.794     -2.996      0.003     -38.626      -8.074
weathersit_3   -59.8136      9.140     -6.544      0.000     -77.729     -41.898
==============================================================================
Omnibus:                     3431.517   Durbin-Watson:                   0.558
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             6705.723
Skew:                           1.201   Prob(JB):                         0.00
Kurtosis:                       4.869   Cond. No.                         784.
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

'''