# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 12:22:20 2020

@author: Bharat Mishra, Patricia Perez-Martin, Pinelopi Christodoulou
"""
# %% Import all necessary Python libraries (Neural Network Implementation using Keras alone)

from keras.models import Sequential     #This allows appending layers to existing models
from keras.layers import Dense          #This allows defining the characteristics of a particular layer
from keras import regularizers          #This allows using whichever regularizer we want (l1,l2,l1_l2)
from keras import optimizers            #This allows using whichever optimiser we want (sgd,adam,RMSprop)
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score

from sklearn.datasets import load_breast_cancer

# close all previous images
plt.close('all')

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# ensure the same random numbers appear every time
np.random.seed(0)


# %% BREAST CANCER DATASET
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

# SCALE DATA

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)



# %% NEURAL NETWORK : TUNABLE PARAMETERS

"""Define tunable model parameters"""

eta=np.logspace(-3,-1,3)           #Define vector of learning rates (parameter to SGD optimiser)
lamda=np.logspace(-4,-2,3)         #Define vector of hyperparameters 
n_layers=2                         #Define number of hidden layers in the model
#n_neuron=(x_train.shape[1])       #Define number of neurons per layer in the model (L here for simplicity)
n_neuron=128
epochs=50                         #Number of reiterations over the input data
batch_size=50                     #Number of samples per gradient update

# %% NEURAL NETWORK : FUNCTIONS

"""Define custom metric"""

def R2_score(y_true,y_pred):
    SS_res=K.sum(K.square(y_true-y_pred)) 
    SS_tot=K.sum(K.square(y_true-K.mean(y_pred))) 
    return ( 1 - SS_res/(SS_tot) )

"""Define function to create Deep Neural Network Model using Keras"""

def NN_Model(n_layers,n_neuron,eta,lamda):
    model=Sequential()
    for i in range(n_layers):              #Run loop to add hidden layers to model
        if (i==0):                         #First layer requires input dimensions = 40 spins
            model.add(Dense(n_neuron,activation='relu',kernel_regularizer=regularizers.l2(lamda),input_dim=40))
        else:                              #Corresponding layers are capable of automatic shape inference
            model.add(Dense(n_neuron,activation='relu',kernel_regularizer=regularizers.l2(lamda))) #Using l2
    model.add(Dense(1,activation='relu'))      #add output layer to model
    sgd=optimizers.SGD(lr=eta)    #Define optimiser for model (lr is learning rate)
    model.compile(loss='mean_squared_error',optimizer=sgd)   #can add metric here, but not necessary
    return model
    
# %%
    
"""Perform regression on data"""

R2_train=np.zeros((len(lamda),len(eta)))     #Define vector to store R2 metric for training data
R2_test=np.zeros((len(lamda),len(eta)))      #Define vector to store R2 metric for testing data

for i in range(len(lamda)):                  #Run loops over hyperparamaters and learning rates
    for j in range(len(eta)):
        DNN_model=NN_Model(n_layers,n_neuron,eta[j],lamda[i])   #Call model for each lamda and eta
        DNN_model.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,verbose=1)  #training on data
        y_train_pred=DNN_model.predict(X_train)       #predict y_train values
        y_test_pred=DNN_model.predict(X_test)         #predict y_test values
        R2_train[i,j]=r2_score(y_train,y_train_pred)  #calculate R2 scores 
        R2_test[i,j]=r2_score(y_test,y_test_pred)
        
# %% 
        
"""Plot results (no Bootstrap framework)"""

plt.figure()
plt.semilogx(lamda,R2_train[:,0],'--g',label='Learning Rate=1E-3, Train Data')
plt.semilogx(lamda,R2_test[:,0],'-*g',label='Learning Rate=1E-3, Test Data')
plt.semilogx(lamda,R2_train[:,1],'--r',label='Learning Rate=1E-2, Train Data')
plt.semilogx(lamda,R2_test[:,1],'-*r',label='Learning Rate=1E-2, Test Data')
plt.semilogx(lamda,R2_train[:,2],'--b',label='Learning Rate=1E-1, Train Data')
plt.semilogx(lamda,R2_test[:,2],'-*b',label='Learning Rate=1E-1, Test Data')
plt.legend()
plt.xlabel('Hyperparameters')
plt.ylabel('R2')
plt.title('Deep Neural Network Performance')
plt.show()








