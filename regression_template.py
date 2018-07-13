
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
#you have to first set upwork directory

#the following split data set into independent variable and dependent variable
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values #the upper bound is not included. we ALWAYS want x to be matrix
y = dataset.iloc[:, 2].values

#split the dataset into training set and test set
#the following split the data set into training and testing, testing 20%
'''from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
'''
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
poly_regressor = PolynomialFeatures(degree = 4)
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

#visualizing the polynomial regression results
#x_grid will contain all the levels and the incremental numbers
x_grid = np.arange(min(x), max(x), 0.1)
#create a matrix, len(x_grid) is the rows, and 1 is the columns
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color='red')
plt.plot(x_grid, linear_regressor_2.predict(poly_regressor.fit_transform(x_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#predicting a new result with Linear Regression
linear_regressor.predict(6.5)

#predicting a new result with polynomial regression
linear_regressor_2.predict(poly_regressor.fit_transform(6.5))






























