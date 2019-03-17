#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:42:28 2019

@author: rafay
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import numpy as np
import os

print(os.getcwd())
os.chdir("/Users/rafay/Pywork/justPractice/KaggleCompetition") #Replace working directory as your own

#Import Train and Test files
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#Get x and y values from teh dataset
X = train["budget"].to_frame() #x is the budget(training)

y = train["revenue"].to_frame() #y is the revenue(training)

#Splitting the training set further into a training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

#Fit the data
regressor = LinearRegression()
model = regressor.fit(X_train,y_train)

#Prediction of revenue
y_pred = regressor.predict(X_test)


#Root Mean Squared Logarithmic Error
from sklearn.metrics import mean_squared_log_error
from math import sqrt
rmsle = sqrt(mean_squared_log_error(y_test, y_pred))
print(rmsle)

#Root Mean Squared Error
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(y_test, y_pred))


#Visualize the Training results
import matplotlib.pyplot as plt
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Budget vs Revenue(Training)")
plt.xlabel("Budget")
plt.ylabel("Revenue")
plt.show()


#Visualize the Test results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Budget vs Revenue(Test)")
plt.xlabel("Budget")
plt.ylabel("Revenue")
plt.show()


#R Squared
r2 = model.score(X_train, y_train)

#Model Summary
X = sm.add_constant(X)

my_model = sm.OLS(y, X)
result = my_model.fit()

print(result.summary())



#Making Predictions on the Test set(Original Test set)
X_X = test["budget"].to_frame()
test_y_pred = regressor.predict(X_X)

#Export our results to a submission .csv file 
submission_file = pd.read_csv('sample_submission.csv')
submission_file['revenue'] = test_y_pred
submission_file.to_csv('submission_file.csv', index=False)











#practice snippets
#train = train['runtime'].fillna(0) #Fill NAs
#X = train.iloc[:,2] #X
#y = train.iloc[:,22].values #y
#submission_file.head() #get first few cols and rows of the file