#data preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
#you have to first set upwork directory

#the following split data set into independent variable and dependent variable
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#split the dataset into training set and test set
#the following split the data set into training and testing, testing 20%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

'''
#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
#need to fit it and then transform it, but if fit the x_train it will then fit the x_test
#so no need to fit the x_test again
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
'''

#fitting simple linear regression to the training set
#import linear regression classs
from sklearn.linear_model import LinearRegression
#create the object
regressor = LinearRegression()





















































