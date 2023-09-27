# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 21:00:12 2023

@author: Salma
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#to ignore warnings
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("job_descriptions.csv")
data.info()
print(data.nunique())


print(data.isnull().sum())
print(data.describe().T)
print(data.describe(include='all').T)
""" display informaion """
"""
print(data.head())
print(data.tail())



"""

"""


"""
"""  separate Numerical and categorical variables for easy analysis """


cat_cols=data.select_dtypes(include=['object']).columns
num_cols = data.select_dtypes(include=np.number).columns.tolist()
print("Categorical Variables:")
print(cat_cols)
print("Numerical Variables:")
print(num_cols)

"""  show the pattern of the variables """


for col in num_cols:
        print(col)
        print('Skew :', round(data[col].skew(), 2))
        plt.figure(figsize = (15, 4))
        plt.subplot(1, 2, 1)
        data[col].hist(grid=False)
        plt.ylabel('count')
        plt.subplot(1, 2, 2)
        sns.boxplot(x=data[col])
        plt.show() 
    
"""  Log transformation can help in normalization    """
    
"""  
def log_transform(data,col):
    for colname in col:
        if (data[colname] == 1.0).all():
            data[colname + '_log'] = np.log(data[colname]+1)
        else:
            data[colname + '_log'] = np.log(data[colname])
    data.info()

log_transform(data,['chol','age'])

sns.distplot(data["chol_log"], axlabel="chol_log");

plt.figure(figsize=(13,17))
sns.pairplot(data=data.drop(['chol','age'],axis=1))
plt.show()

"""
