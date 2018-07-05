#data preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#import the dataset
#you have to first set upwork directory

#the following split data set into independent variable and dependent variable
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#how to transform categorical data
LabelEncoder_x = LabelEncoder()
x[:, 3] = LabelEncoder_x.fit_transform(x[:, 3]) #this line means transform the category text into 1,2,3

#dummy encoder, transform countries into several columns
#but you need to transform them in the above state first to numbers, and then split into 1s
onhotencoder = OneHotEncoder(categorical_features=[3])
x = onhotencoder.fit_transform(x).toarray()

#avoid the dummy variable trap, delete 1 variable
x = x[:, 1:]

#split the dataset into training set and test set
#the following split the data set into training and testing, testing 20%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#fitting multiple linear regressions to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predicting the test set result, use x_test to see the predicted y
y_pred = regressor.predict(x_test)

#build the optimal model using backward elimination
#import statsmodels.formula.api as sm

#append X0 as 1s to add the b0, the function always append number to the end
x = np.append(arr = np.ones((50, 1)).astype(int), values = x, axis=1)























