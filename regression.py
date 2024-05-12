import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv('C:\\Users\\bekta\\PycharmProjects\\staj\\Salary Data.csv')
#Dosyanın ilk 5 elemanını yazdırır.
print(df.head())
df.info()

#Veri çerçevesindeki sayısal sütunların istatistiksel özetini sağlar.
print(df.describe())
print(df.isnull().sum())

#ValueError: Input contains NaN, infinity or a value too large for dtype('float64')
# hatasını gidermek için alt satırı ekledim.
#qdf =df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

#Data Visualization
plt.scatter(df["Salary"],df["Years of Experience"])
plt.xlabel("Salary")
plt.ylabel("No of Years Experience")
plt.title("Salary vs Years of Experience")
plt.show()

plt.hist(df["Salary"])
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.title("Salary Distribution")
plt.show()

plt.hist(df["Years of Experience"])
plt.xlabel("Years of Experience")
plt.ylabel("Frequency")
plt.title("Years Experience Distribution")
plt.show()

correlation_matrix = df.corr()
print(correlation_matrix)

x = df["Years of Experience"].values
x = x.reshape(-1, 1)

y = df["Salary"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

model = LinearRegression()
print(model.fit(x_train, y_train))

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("mse =", mse)
print("r2 =", r2)
