# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 19:15:04 2021

@author: kavita.gaikwad
"""
### An example of low coefficients in between independent features
import pandas as pd

import statsmodels.api as sm
df_adv = pd.read_csv("Advertising.csv", index_col=0)
X = df_adv[['TV','Radio','Newspaper']]
y = df_adv['Sales']
df_adv.head()


## fit a OLS with intercepts on TV and Radio
X = sm.add_constant(X)
### print X in console and check 

model = sm.OLS(y,X).fit()


model.summary()


import matplotlib.pyplot as plt
X.iloc[:,1:].corr()


### An example of High coefficients in between independent features

df_salary = pd.read_csv("Salary_data.csv")

df_salary.head()

X= df_salary[['YearsExperience','Age']]

y = df_salary[['Salary']]

## Fit a OLS model with intercept 
X= sm.add_constant(X)
model = sm.OLS(y,X).fit()


model.summary()

X.iloc[:,1:].corr()


