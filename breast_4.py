# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 10:00:03 2020

@author: Patricia
"""
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
print("labels =  ")
print (labels)


# flatten the image
# the value -1 means dimension is inferred from the remaining dimensions: 8x8 = 64
n_inputs = len(inputs)
inputs = inputs.reshape(n_inputs, -1)
print('----------')
print("X = (n_inputs, n_features) = " + str(inputs.shape))


# %% TRAIN AND TEST DATASET

# Set up training data: from scikit-learn library
train_size = 0.8
test_size = 1 - train_size
X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, train_size=train_size,
                                                    test_size=test_size)

print("Number of training data: " + str(len(X_train)))
print("Number of test data: " + str(len(X_test)))


# %% LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression

print('----------------------')
print('LOGISTIC REGRESSION')
print('----------------------')
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


"""
# Drawing a correlation graph it is possible to remove multi colinearity, since features
# dependenig on each do not apport new information. 
# 
# Only features_mean are mainly studied in this code.

"""
import pandas as pd
import seaborn as sns 

# Making a data frame
breastpd = pd.DataFrame(inputs, columns=labels)

corr = breastpd.corr().round(1)

# use the heatmap function from seaborn to plot the correlation matrix
plt.figure()
sns.heatmap(corr, cbar = True,  square = True, annot=False,
           xticklabels= labels, yticklabels= labels,
           cmap= 'YlOrRd')

"""
ANALYSIS

- Radius, perimeter and area are highly correlated (as it was expected), so only one of them will be used
- Compactness, concavity and concavepoint are also highly correlated, so we will only use compactness_mean
- Therefore the chosen parameters are perimeter_mean, texture_mean, compactness_mean and symmetry_mean

"""
# %% RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
print('----------------------')
print('RANDOM FOREST')
print('----------------------')
model=RandomForestClassifier(n_estimators=100)# a simple random forest model
model.fit(X_train,y_train)# now fit our model for training data

prediction=model.predict(X_test)# predict for the test data


# ACCURACY
from sklearn import metrics # for the check the error and accuracy of the model

print('Accuracy Random Forest: ')
acc = metrics.accuracy_score(prediction,y_test) # to check the accuracy
print(acc)

"""
observation

Here the Accuracy for our model is 91 % which seems good*
"""
# %% DEFINE MODEL AND ARCHITECTURE: NEURAL NETWORK
print('----------------------')
print('NEURAL NETWORK')
print('----------------------')
# building our neural network

n_inputs, n_features = X_train.shape
n_hidden_neurons = 20
n_categories = 10

# we make the weights normally distributed using numpy.random.randn

# weights and bias in the hidden layer
hidden_weights = np.random.randn(n_features, n_hidden_neurons)
hidden_bias = np.zeros(n_hidden_neurons) + 0.01

# weights and bias in the output layer
output_weights = np.random.randn(n_hidden_neurons, n_categories)
output_bias = np.zeros(n_categories) + 0.01

# %% FEED-FORWARD PASS, subscript h = hidden layer

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def feed_forward(X):
    # weighted sum of inputs to the hidden layer
    z_h = np.matmul(X, hidden_weights) + hidden_bias
    # activation in the hidden layer
    a_h = sigmoid(z_h)
    
    # weighted sum of inputs to the output layer
    z_o = np.matmul(a_h, output_weights) + output_bias
    # softmax output
    # axis 0 holds each input and axis 1 the probabilities of each category
    exp_term = np.exp(z_o)
    probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
    
    return probabilities

probabilities = feed_forward(X_train)
print("probabilities = (n_inputs, n_categories) = " + str(probabilities.shape))
print("probability that image 0 is in category 0,1,2,...,9 = \n" + str(probabilities[0]))
print("probabilities sum up to: " + str(probabilities[0].sum()))
print()

# we obtain a prediction by taking the class with the highest likelihood
def predict(X):
    probabilities = feed_forward(X)
    return np.argmax(probabilities, axis=1)


predictions = predict(X_train)
print("predictions = (n_inputs) = " + str(predictions.shape))
print("prediction for image 0: " + str(predictions[0]))
print("correct label for image 0: " + str(y_train[0]))

# %% OPTIMIZING COST FUNCTION

# to categorical turns our integer vector into a onehot representation
from sklearn.metrics import accuracy_score

# one-hot in numpy
def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    
    return onehot_vector

#Y_train_onehot, Y_test_onehot = to_categorical(Y_train), to_categorical(Y_test)
Y_train_onehot, Y_test_onehot = to_categorical_numpy(y_train), to_categorical_numpy(y_test)

def feed_forward_train(X):
    # weighted sum of inputs to the hidden layer
    z_h = np.matmul(X, hidden_weights) + hidden_bias
    # activation in the hidden layer
    a_h = sigmoid(z_h)
    
    # weighted sum of inputs to the output layer
    z_o = np.matmul(a_h, output_weights) + output_bias
    # softmax output
    # axis 0 holds each input and axis 1 the probabilities of each category
    exp_term = np.exp(z_o)
    probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
    
    # for backpropagation need activations in hidden and output layers
    return a_h, probabilities

def backpropagation(X, Y):
    a_h, probabilities = feed_forward_train(X)
    
    # error in the output layer
    error_output = probabilities - Y
    # error in the hidden layer
    error_hidden = np.matmul(error_output, output_weights.T) * a_h * (1 - a_h)
    
    # gradients for the output layer
    output_weights_gradient = np.matmul(a_h.T, error_output)
    output_bias_gradient = np.sum(error_output, axis=0)
    
    # gradient for the hidden layer
    hidden_weights_gradient = np.matmul(X.T, error_hidden)
    hidden_bias_gradient = np.sum(error_hidden, axis=0)

    return output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient

print("Old accuracy on training data: " + str(accuracy_score(predict(X_train), y_train)))

eta = 0.01
lmbd = 0.01
for i in range(1000):
    # calculate gradients
    dWo, dBo, dWh, dBh = backpropagation(X_train, Y_train_onehot)
    
    # regularization term gradients
    dWo += lmbd * output_weights
    dWh += lmbd * hidden_weights
    
    # update weights and biases
    output_weights -= eta * dWo
    output_bias -= eta * dBo
    hidden_weights -= eta * dWh
    hidden_bias -= eta * dBh

print("New accuracy on training data: " + str(accuracy_score(predict(X_train), y_train)))
