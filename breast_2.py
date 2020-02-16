# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 17:14:36 2020

BASED ON https://towardsdatascience.com/building-a-simple-machine-learning-model-on-breast-cancer-data-eca4b3b99fa3

@author: Patricia
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing our cancer dataset

from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()

X = dataset.iloc[:, 1:31].values
Y = dataset.iloc[:, 31].values

# We can examine the data set using the pandas’ head() method
dataset.head()

print("Cancer data set dimensions : {}".format(dataset.shape))

# Diagnosis column is an object type so we can map it to integer value
dataset['diagnosis']=data['diagnosis'].map({'M':1,'B':0})

# %% Missing or Null Data points

dataset.isnull().sum()
dataset.isna().sum()

# %% LOGISTIC REGRESSION : simple regression case on the breast cancer data using logistic regression as algorithm for classification

from sklearn.model_selection import  train_test_split 
from sklearn.linear_model import LogisticRegression
#cancer = load_breast_cancer()

# Set up training data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
print("Test set accuracy: {:.2f}".format(logreg.score(X_test,Y_test)))

# Scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
logreg.fit(X_train_scaled, Y_train)
print("Test set accuracy scaled data: {:.2f}".format(logreg.score(X_test_scaled,Y_test)))

# %% COVARIANCE AND CORRELATION
# Making a data frame
cancerpd = pd.DataFrame(cancer.data, columns=cancer.feature_names)

fig, axes = plt.subplots(15,2,figsize=(10,20))
malignant = cancer.data[cancer.target == 0]
benign = cancer.data[cancer.target == 1]
ax = axes.ravel()

for i in range(30):
    _, bins = np.histogram(cancer.data[:,i], bins =50)
    ax[i].hist(malignant[:,i], bins = bins, alpha = 0.5)
    ax[i].hist(benign[:,i], bins = bins, alpha = 0.5)
    ax[i].set_title(cancer.feature_names[i])
    ax[i].set_yticks(())
ax[0].set_xlabel("Feature magnitude")
ax[0].set_ylabel("Frequency")
ax[0].legend(["Malignant", "Benign"], loc ="best")
fig.tight_layout()
plt.show()

import seaborn as sns
correlation_matrix = cancerpd.corr().round(1)
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
Y = labelencoder_Y.fit_transform(Y)

# %% SPLITTING DATASET

""" Splitting the dataset
The data we use is usually split into training data and test data. The training set
 contains a known output and the model learns on this data in order to be generalized
 to other data later on. We have the test dataset (or subset) in order to test our
 model’s prediction on this subset.
 
We will do this using SciKit-Learn library in Python using the train_test_split method.

"""

# Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# %% SCALING

"""
Phase 3 — Feature Scaling
Most of the times, your dataset will contain features highly varying in magnitudes,
 units and range. But since, most of the machine learning algorithms use Eucledian distance
 between two data points in their computations. We need to bring all features to the same level
 of magnitudes. This can be achieved by scaling. This means that you’re transforming your data so
 that it fits within a specific scale, like 0–100 or 0–1.
 
We will use StandardScaler method from SciKit-Learn library.
"""

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# %% MODEL SELECTION

"""
This is the most exciting phase in Applying Machine Learning to any Dataset. It 
is also known as Algorithm selection for Predicting the best results.

Usually Data Scientists use different kinds of Machine Learning algorithms to the 
large data sets. But, at high level all those different algorithms can be classified 
in two groups : supervised learning and unsupervised learning.

Without wasting much time, I would just give a brief overview about these two types of learnings.
Supervised learning : Supervised learning is a type of system in which both input 
and desired output data are provided. Input and output data are labelled for 
classification to provide a learning basis for future data processing. Supervised 
learning problems can be further grouped into Regression and Classification problems.

A regression problem is when the output variable is a real or continuous value, such as “salary” or “weight”.
A classification problem is when the output variable is a category like filtering emails “spam” or “not spam”

Unsupervised Learning : Unsupervised learning is the algorithm using information 
that is neither classified nor labeled and allowing the algorithm to act on that 
information without guidance.

In our dataset we have the outcome variable or Dependent variable i.e Y having only 
two set of values, either M (Malign) or B(Benign). So we will use Classification
 algorithm of supervised learning.
 
We have different types of classification algorithms in Machine Learning :

1. Logistic Regression
2. Nearest Neighbor
3. Support Vector Machines
4. Kernel SVM
5. Naïve Bayes
6. Decision Tree Algorithm
7. Random Forest Classification
Lets start applying the algorithms :
We will use sklearn library to import all the methods of classification algorithms.
We will use LogisticRegression method of model selection to use Logistic Regression Algorithm,    
 


"""

#Using Logistic Regression Algorithm to the Training Set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)
#Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, Y_train)
#Using SVC method of svm class to use Support Vector Machine Algorithm
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, Y_train)
#Using SVC method of svm class to use Kernel SVM Algorithm
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, Y_train)
#Using GaussianNB method of naïve_bayes class to use Naïve Bayes Algorithm
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)
#Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)

#Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)


# %% PREDICTION AND ACCURACY

# We will now predict the test set results and check the accuracy with each of our model:
Y_pred = classifier.predict(X_test)

"""
To check the accuracy we need to import confusion_matrix method
 of metrics class. The confusion matrix is a way of tabulating the
 number of mis-classifications, i.e., the number of predicted classes which
 ended up in a wrong classification bin based on the true classes.
 
"""

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

"""
We will use Classification Accuracy method to find the accuracy of our models.
 Classification Accuracy is what we usually mean, when we use the term accuracy. 
 It is the ratio of number of correct predictions to the total number of input samples.


To check the correct prediction we have to check confusion matrix object and
 add the predicted results diagonally which will be number of correct prediction
 and then divide by total number of predictions.
 
 
After applying the different classification models, we have got below accuracies with different models:
1. Logistic Regression — 95.8%
2. Nearest Neighbor — 95.1%
3. Support Vector Machines — 97.2%
4. Kernel SVM — 96.5%
5. Naive Bayes — 91.6%
6. Decision Tree Algorithm — 95.8%
7. Random Forest Classification — 98.6%
So finally we have built our classification model and we can see that Random Forest Classification algorithm gives the best results for our dataset. Well its not always applicable to every dataset. To choose our model we always need to analyze our dataset and then apply our machine learning model.
This is a basic application of Machine Learning Model to any dataset 

"""