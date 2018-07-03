#data preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
#you have to first set upwork directory

#the following split data set into independent variable and dependent variable
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#the following fill the null with mean data, to make every line valuable
#fill in the missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(x[:, 1:3]) #the first '#' means all the lines, the second means 2 to 4 columns
x[:, 1:3] = imputer.transform(x[:, 1:3]) #replace x the part with the new part, transform means replace

'''
Like other estimators, these are represented by classes with a fit method, which learns model 
parameters (e.g. mean and standard deviation for normalization) from a training set, and a transform 
method which applies this transformation model to unseen data. fit_transform may be more convenient and 
efficient for modelling and transforming the training data simultaneously.
'''

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#to transform the country names into numbers 1,2,3
#how to transform categorical data
LabelEncoder_x = LabelEncoder()
x[:, 0] = LabelEncoder_x.fit_transform(x[:, 0]) #this line means transform the category text into 1,2,3

#dummy encoder, transform countries into several columns
#but you need to transform them in the above state first to numbers, and then split into 1s
onhotencoder = OneHotEncoder(categorical_features=[0])
x = onhotencoder.fit_transform(x).toarray()

#the following labelEncode the y which is purcahsed or not
#purchased or not is only 0, 1, so directly transform don't need toarray
LabelEncoder_y = LabelEncoder()
y = LabelEncoder_y.fit_transform(y)

#split the dataset into training set and test set
#the following split the data set into training and testing, testing 20%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#feature scaling
#this part MOST OF THE TIME is included in the ML library
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
#need to fit it and then transform it, but if fit the x_train it will then fit the x_test
#so no need to fit the x_test again
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


























