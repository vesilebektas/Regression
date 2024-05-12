import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

data = pd.read_csv('C:\\Users\\bekta\\PycharmProjects\\staj\\Salary_Data.csv')
# print(data.head())
#
# data.info()
# print(data.describe())
#
# plt.figure(figsize=(12,6))
# sns.pairplot(data,x_vars= ['YearsExperience'],y_vars=['Salary'],height=7,kind='scatter')
# plt.xlabel('Years')
# plt.ylabel('Salary')
# plt.title('Salary Prediction')
# plt.show()
#
X = data['YearsExperience']
print(X.head())

y = data['Salary']
print(y.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7,random_state = 100)

X_train = X_train[:,np.newaxis]
X_test = X_test[:,np.newaxis]

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
print(lr.fit(X_train,y_train))

y_pred= lr.predict(X_test)

c = [i for i in range (1,len(y_test)+1,1)]
plt.plot(c,y_test,color='r',linestyle = '-')
plt.plot(c,y_pred,color='b',linestyle='-')
plt.xlabel('Salary')
plt.ylabel('index')
plt.title('Prediction')
plt.show()

c = [i for i in range(1, len(y_test)+1, 1)]
plt.plot(c, y_test - y_pred, color='green', linestyle='-')
plt.xlabel('index')
plt.ylabel('Error')
plt.title('Error Value')
plt.show()

from sklearn.metrics import r2_score,mean_squared_error

mse = mean_squared_error(y_test,y_pred)
rsq = r2_score(y_test,y_pred)

print('mean squared error:' ,mse)
print('r square:', rsq)

plt.figure(figsize=(12,6))
plt.scatter(y_test, y_pred, color  = 'r',linestyle='-')
plt.show()

print('Intercept of the model:', lr.intercept_)
print('Coefficient of the line:', lr.coef_)
