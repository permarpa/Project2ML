# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 13:34:20 2020

@author: Patricia
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import  train_test_split 
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import seaborn as sns 
from sklearn.model_selection import KFold # use for cross validation
from sklearn.model_selection import GridSearchCV# for tuning parameter
from sklearn.ensemble import RandomForestClassifier # for random forest classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm # for Support Vector Machine
from sklearn import metrics # for the check the error and accuracy of the model
dataset = load_breast_cancer()


plt.close('all')

# %% PREPROCESSING DATA
list(dataset.target_names)
X, y = dataset.data, dataset.target

#list(dataset.feature_names)

# Data can be divided into three parts, according to their category
features_mean= dataset.feature_names[0:10]
features_se= list(dataset.feature_names[10:20])
features_worst=list(dataset.feature_names[20:30])
print(features_mean)
print("-----------------------------------")
print(features_se)
print("------------------------------------")
print(features_worst)



# %% DATA ANALYSIS: A LITTLE FEATURE SELECTION

# now lets draw a correlation graph so that we can remove multi colinearity it means the columns are
# dependenig on each other so we should avoid it because what is the use of using same column twice
# lets check the correlation between features
# now we will do this analysis only for features_mean then we will do for others and will see who is doing best


import pandas as pd
# Making a data frame
breastpd = pd.DataFrame(X, columns=dataset.feature_names)

corr = breastpd.corr().round(1)
# use the heatmap function from seaborn to plot the correlation matrix

plt.figure()
sns.heatmap(corr, cbar = True,  square = True, annot=False,
           xticklabels= dataset.feature_names, yticklabels= dataset.feature_names,
           cmap= 'YlOrRd') # for more on heatmap you can visit Link(http://seaborn.pydata.org/generated/seaborn.heatmap.html)

"""
observation

- the radius, parameter and area are highly correlated as expected from their relation so from these we will use anyone of them
 compactness_mean, concavity_mean and concavepoint_mean are highly correlated so we will use compactness_mean from here
- so selected Parameter for use is perimeter_mean, texture_mean, compactness_mean, symmetry_mean*

"""
# Variables which will use for prediction
prediction_var = ['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean']

print ('Prediction Variables')
print (prediction_var)

# %% TRAIN AND TEST DATA: only for mean perimeter

#now split our data into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# we can check their dimension
print('-----------------')
print(X_train.shape)
print(X_test.shape)
print('-----------------')

#
X_train_per = X_train[:, np.newaxis, 2]# taking the training data input 
#train_y=y_train.diagnosis       # This is output of our training data
## same we have to do for test
X_test_per= X_test[:, np.newaxis, 2]  # taking test data inputs
#test_y =y_test.diagnosis        #output value of test dat
#
print(X_train_per )
print('-----------------')
print(X_test_per )
print('-----------------')


# %% LINEAR REGRESSION

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)


# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# Plot outputs
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()



# %% LOGISTIC REGRESSION 
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print("Test set accuracy: {:.2f}".format(logreg.score(X_test,y_test)))

# Scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
logreg.fit(X_train_scaled, y_train)
print("Test set accuracy scaled data: {:.2f}".format(logreg.score(X_test_scaled,y_test)))

# %% COVARIANCE AND CORRELATION 

    #import pandas as pd
    ## Making a data frame
    #breastpd = pd.DataFrame(X, columns=dataset.feature_names)

fig, axes = plt.subplots(15,2,figsize=(10,20))
malignant = X[y == 0]
benign = X[y == 1]
ax = axes.ravel()

for i in range(30):
    _, bins = np.histogram(X[:,i], bins =50)
    ax[i].hist(malignant[:,i], bins = bins, alpha = 0.5)
    ax[i].hist(benign[:,i], bins = bins, alpha = 0.5)
    ax[i].set_title(dataset.feature_names[i])
    ax[i].set_yticks(())
ax[0].set_xlabel("Feature magnitude")
ax[0].set_ylabel("Frequency")
ax[0].legend(["Malignant", "Benign"], loc ="best")
fig.tight_layout()
plt.show()

import seaborn as sns
correlation_matrix = breastpd.corr().round(1)
# use the heatmap function from seaborn to plot the correlation matrix
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()

#print eigvalues of correlation matrix
EigValues, EigVectors = np.linalg.eig(correlation_matrix)
print(EigValues)

# %% CATEGORICAL DATA

""" 
Categorical data are variables that contain label values rather than numeric values.
The number of possible values is often limited to a fixed set.
For example, users are typically described by country, gender, age group etc.
We will use Label Encoder to label the categorical data. Label Encoder is the part
of SciKit Learn library in Python and used to convert categorical data, or text data,
into numbers, which our predictive models can better understand.
"""

#Encoding categorical data values
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)

# %% RANDOM FOREST

model=RandomForestClassifier(n_estimators=100)# a simple random forest model
model.fit(train_X,train_y)# now fit our model for traiing data

prediction=model.predict(test_X)# predict for the test data
# prediction will contain the predicted value by our model predicted values of dignosis column for test inputs

# %% ACCURACY

print('Accuracy Random Forest: ')
metrics.accuracy_score(prediction,test_y) # to check the accuracy
# here we will use accuracy measurement between our predicted value and our test output values

"""
observation

Here the Accuracy for our model is 91 % which seems good*
"""