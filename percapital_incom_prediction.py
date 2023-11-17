import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

## read the data using pandas
data = pd.read_csv('canada_per_capita_income.csv')
print(data)

## to display the scatter plot 
# plt.xlabel("Years")
# plt.ylabel("Per capital income")
# plt.scatter(data["year"],data["per capita income (US$)"],color='red',marker='+')
# plt.show()

## training the model

reg1 = linear_model.LinearRegression()
reg1.fit(data[["year"]],data[["per capita income (US$)"]])
predict = reg1.predict(data[["year"]])
data["pred"] = predict
print(data)

## to predict the year 2020

val = reg1.predict([[2020]])
print(val)