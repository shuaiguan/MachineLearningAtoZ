
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

#fitting regression model to the dataset
#create the regressor

#predicting a new result with polynomial regression
y_pred = regressor.predict(6.5)


#visualizing the polynomial regression results
#x_grid will contain all the levels and the incremental numbers
#this step to add more resolution for the visualization of predictions
x_grid = np.arange(min(x), max(x), 0.1)
#create a matrix, len(x_grid) is the rows, and 1 is the columns
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color='red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()




























