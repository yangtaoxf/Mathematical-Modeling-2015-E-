import pandas as pd
import statsmodels.api as sm

filename = "data.csv"
data = pd.read_csv(filename)

x = data.ix[:,1:data.shape[1]-1]
y = data.ix[:,data.shape[1]-1]

y = y.values.reshape((y.shape[0],1))

model = sm.OLS(y,x)
result = model.fit()
print(result.params)
print(result.summary())
