
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
#you have to first set upwork directory

#the following split data set into independent variable and dependent variable
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values #the upper bound is not included. we ALWAYS want x to be matrix
y = dataset.iloc[:, 2].values

#don't need to split the dataset because the dataset is really small...

'''
#feature scaling
we don't need feature scaling because polynomial 
'''

#first build linear regression to see the result to compare with polynomial
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(x, y)

#fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
#degree means the power of the variable
poly_regressor = PolynomialFeatures(degree = 2)
#need to fit and then transform because it will transform the x
x_poly = poly_regressor.fit_transform(x)
linear_regressor_2 = LinearRegression()
linear_regressor_2.fit(x_poly, y)

#visualizing the linear regression results
plt.scatter(x, y, color='red')
plt.plot(x, linear_regressor.predict(x), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

















