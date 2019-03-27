#%%
import pandas as pd 
import quandl

df = quandl.get("WIKI/GOOGL")
print(df.head())
print(df.shape)
#print(df.head())
#df = df[['Adj.Open',  'Adj.High',  'Adj.Low',  'Adj.Close', 'Adj.Volume']]
#df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df = df[['Adj. Open',  'Adj. High',  'Adj. Low', 'Adj. Close', 'Adj. Volume']]
print(df.shape)
print(df)

df['HL_PCT'] = ((df['Adj. High']-df['Adj. Low'])/df['Adj. Close'])*100
df['PCT_change'] = ((df['Adj. Close']-df['Adj. Low'])/df['Adj. Open'])*100
print(df)
#%%
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
print(df.head())
print(df.shape)

#df = df[[]]

#%%
import numpy as np 
import pandas as pd
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import math

forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))
df['label']= df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

print(df.head())

X = np.array(df.drop(['label'],1))
y= np.array(df['label'])

X = preprocessing.scale(X)


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .25)



#%%
#Implementing svm
clf = svm.SVR()
clf.fit(X_train,y_train)

confidence = clf.score(X_test,y_test)
print(confidence)





#%%

#Implementing linear regression

clf = LinearRegression()
clf.fit(X_train,y_train)
confidence = clf.score(X_test,y_test)
print(confidence)

#%%

