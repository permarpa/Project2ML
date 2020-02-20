# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 10:00:03 2020

@author: Bharat Mishra, Patricia Perez-Martin, Pinelopi Christodoulou
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split 
from sklearn.datasets import load_breast_cancer

# close all previous images
plt.close('all')

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# ensure the same random numbers appear every time
np.random.seed(0)

# download breast cancer dataset
cancer = load_breast_cancer()

# define inputs and labels
inputs = cancer.data
outputs = cancer.target     #Malignant or bening
labels = cancer.feature_names[0:30]


print('The content of the breast cancer dataset is:')
print(labels)
print('-------------------------')
print("inputs =  " + str(inputs.shape))
print("outputs =  " + str(outputs.shape))
print("labels =  "+ str(labels.shape))


# flatten the image
# the value -1 means dimension is inferred from the remaining dimensions: 8x8 = 64
n_inputs = len(inputs)
#inputs = inputs.reshape(n_inputs, -1)
print('----------')
print("X = (n_inputs, n_features) = " + str(inputs.shape))

#%% VISUALIZATION

X = inputs
y = outputs


plt.figure()
plt.scatter(X[:,0], X[:,2], s=40, c=y, cmap=plt.cm.Spectral)
plt.xlabel('Mean radius')
plt.ylabel('Mean perimeter')
plt.show()

plt.figure()
plt.scatter(X[:,5], X[:,6], s=40, c=y, cmap=plt.cm.Spectral)
plt.xlabel('Mean compactness')
plt.ylabel('Mean concavity')
plt.show()


plt.figure()
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
plt.xlabel('Mean radius')
plt.ylabel('Mean texture')
plt.show()

plt.figure()
plt.scatter(X[:,2], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
plt.xlabel('Mean perimeter')
plt.ylabel('Mean compactness')
plt.show()


# %% COVARIANCE AND CORRELATION 

import pandas as pd
import seaborn as sns 

# Making a data frame
breastpd = pd.DataFrame(inputs, columns=labels)

corr = breastpd.corr().round(1)		# Compute pairwise correlation of columns, excluding NA/null values.

# use the heatmap function from seaborn to plot the correlation matrix
plt.figure()
sns.heatmap(corr, cbar = True,  square = True, annot=False,
           xticklabels= labels, yticklabels= labels,
           cmap= 'YlOrRd')

# %% 
from sklearn import  linear_model

X_t = X[ : , 1:3]

clf = linear_model.LogisticRegressionCV()
clf.fit(X_t, y)


# Set min and max values and give it some padding
x_min, x_max = X_t[:, 1].min() - .5, X_t[:, 1].max() + .5
y_min, y_max = X_t[:, 0].min() - .5, X_t[:, 0].max() + .5
h = 0.01
# Generate a grid of points with distance h between them
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Predict the function value for the whole gid
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the contour and training examples
plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
plt.scatter(X[:, 2], X[:, 1], c=y, cmap=plt.cm.Spectral)
plt.xlabel('Mean perimeter')
plt.ylabel('Mean texture')
plt.title('Logistic Regression')
plt.show()

# %% TRAIN AND TEST DATASET

# Set up training data: from scikit-learn library
train_size = 0.9
test_size = 1 - train_size

X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, train_size=train_size,
                                                    test_size=test_size)

print("Number of training data: " + str(len(X_train)))
print("Number of test data: " + str(len(X_test)))


# %% LOGISTIC REGRESSION and ACCURACY
from sklearn.linear_model import LogisticRegression


print('----------------------')
print('LOGISTIC REGRESSION')
print('----------------------')
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print("Test set accuracy with Logistic Regression:: {:.2f}".format(logreg.score(X_test,y_test)))

# Scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
logreg.fit(X_train_scaled, y_train)
print("Test set accuracy scaled data: {:.2f}".format(logreg.score(X_test_scaled,y_test)))


# %% DECISION TREES: REGRESSION AND ACCURACY

print('----------------------')
print('DECISION TREES')
print('----------------------')

import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from pydot import graph_from_dot_data
import pandas as pd

#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
regr_1=DecisionTreeRegressor(max_depth=2)
regr_2=DecisionTreeRegressor(max_depth=5)
regr_3=DecisionTreeRegressor(max_depth=11)
regr_1.fit(X, y)
regr_2.fit(X, y)
regr_3.fit(X, y)

y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)
y_3=regr_3.predict(X_test)

# Plot the results
plt.figure()
plt.scatter(X, y, s=2.5, c="black", label="data")
plt.plot(X_test, y_1, color="red",
         label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="green", label="max_depth=5", linewidth=2)
plt.plot(X_test, y_3, color="m", label="max_depth=7", linewidth=2)

plt.xlabel("Data")
plt.ylabel("Target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()


# %% DECISION TREES: CLASSIFICATION and ACCURACY

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from IPython.display import Image 
from pydot import graph_from_dot_data


# Create the encoder.
encoder = OneHotEncoder(handle_unknown="ignore")
# Assume for simplicity all features are categorical.
encoder.fit(X)    
# Apply the encoder.
X = encoder.transform(X)


# Then do a Classification tree

tree_clf = DecisionTreeClassifier(max_depth=None)
tree2_clf = DecisionTreeClassifier(max_depth=2)
tree5_clf = DecisionTreeClassifier(max_depth=5)
tree11_clf = DecisionTreeClassifier(max_depth=11)

tree_clf.fit(X_train, y_train)
tree2_clf.fit(X_train, y_train)
tree5_clf.fit(X_train, y_train)
tree11_clf.fit(X_train, y_train)

test_acc = [tree_clf.score(X_test,y_test),tree2_clf.score(X_test,y_test),tree5_clf.score(X_test,y_test),tree11_clf.score(X_test,y_test)]

print("Test set accuracy with Decision Tree (No Max depth): {:.2f}".format(tree_clf.score(X_test,y_test)))
print("Test set accuracy with Decision Tree (Max depth 2): {:.2f}".format(tree2_clf.score(X_test,y_test)))
print("Test set accuracy with Decision Tree (Max depth 5): {:.2f}".format(tree5_clf.score(X_test,y_test)))
print("Test set accuracy with Decision Tree (Max depth 11): {:.2f}".format(tree11_clf.score(X_test,y_test)))

#transfer to a decision tree graph
export_graphviz(
    tree2_clf,
    out_file="ride.dot",
    rounded=True,
    filled=True
)
cmd = 'dot -Tpng ride.dot -o DecisionTree_max_depth_2.png'
os.system(cmd)

# %% RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
print('----------------------')
print('RANDOM FOREST')
print('----------------------')
model=RandomForestClassifier(n_estimators=100) # a simple random forest model
model.fit(X_train,y_train) # now fit our model for training data

prediction=model.predict(X_test)# predict for the test data


# ACCURACY
from sklearn import metrics # for the check the error and accuracy of the model

print('Accuracy Random Forest: ')
acc = metrics.accuracy_score(prediction,y_test) # to check the accuracy
print(acc)

