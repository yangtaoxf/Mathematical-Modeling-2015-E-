import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from scipy import stats
import statsmodels.graphics.api as smg
import matplotlib.pyplot as plt

# Data with normalization
filename = "data.csv"
data = pd.read_csv(filename)

x = data.ix[:,1:data.shape[1]-1]
y = data.ix[:,data.shape[1]-1]

# Divide by y
#x = x.T
#x = (x/y).T

n_components = 4
pca = PCA(n_components = 4)
pcaX = pca.fit_transform(x)

#print(pca.explained_variance_)
#print(pca.explained_variance_ratio_)

################################
# OLS and add constant
# Shit idea!!!
###############################
#feature_idxs = [9,1,6,2]
#x = data.ix[:,feature_idxs]


pcaX = sm.add_constant(pcaX)
y = y.values.reshape((y.shape[0],1))
model = sm.OLS(y,pcaX)
result = model.fit()
print(result.params)
print(result.summary())

# Resid check
print(result.resid)
z,p = stats.normaltest(result.resid.values)

# QQ
fig, ax = plt.subplots(figsize=(8, 4))
smg.qqplot(result.resid, ax=ax)
plt.show()
