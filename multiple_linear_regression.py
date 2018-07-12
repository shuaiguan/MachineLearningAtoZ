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
import statsmodels.formula.api as sm

#append X0 as 1s to add the b0, the function always append number to the end
#so that the model would understand aX0 + bX1 + cX2 and if X0 is 0 then that is a
x = np.append(arr = np.ones((50, 1)).astype(int), values = x, axis=1)


#to fit the full model with all possible predictors
x_opt = x[:, [0,1,2,3,4,5]]
#need to create another regressor because using OLS class from statsmodel
#class input: endog: the dependent variable, exog: x array, k is the number of regressors
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()

#then consider the predictor with the highest Pvalue, if P>significant value which now is 0.05,
#then move forward, if not then it's done
#the lower the P value is, the more important the variable is
regressor_OLS.summary()

#then remove the highest Pvalue x and then try one more time, in this case it's X2
#be mindful that in the regressor report const is X0
x_opt = x[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()



#then one more time... this time X1
x_opt = x[:, [0,3,5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()


#then one more time.
x_opt = x[:, [0,3]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regressor_OLS.summary()

#create a loop to do so
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x


SL = 0.05
X_opt = x[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

































