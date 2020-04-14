# -*- coding: utf-8 -*-
"""
==================================================================
== Trains a classification model to predict hand-written digits ==
==================================================================
Created on Thu Apr  2 22:48:39 2020
@author: Malandrakis Angelos
"""


#%% Import Libraries & Data

import numpy as np
from mnist import MNIST
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.ndimage import interpolation
from joblib import dump

# Define Deskewing functions
def moments(image):
    c0,c1 = np.mgrid[:image.shape[0],:image.shape[1]] # A trick in numPy to create a mesh grid
    totalImage = np.sum(image) #sum of pixels
    m0 = np.sum(c0*image)/totalImage #mu_x
    m1 = np.sum(c1*image)/totalImage #mu_y
    m00 = np.sum((c0-m0)**2*image)/totalImage #var(x)
    m11 = np.sum((c1-m1)**2*image)/totalImage #var(y)
    m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage #covariance(x,y)
    mu_vector = np.array([m0,m1]) # Notice that these are \mu_x, \mu_y respectively
    covariance_matrix = np.array([[m00,m01],[m01,m11]]) # Do you see a similarity between the covariance matrix
    return mu_vector, covariance_matrix

def deskew(image):
    c,v = moments(image)
    alpha = v[0,1]/v[0,0]
    affine = np.array([[1,0],[alpha,1]])
    ocenter = np.array(image.shape)/2.0
    offset = c-np.dot(affine,ocenter)
    return interpolation.affine_transform(image,affine,offset=offset)



mndata = MNIST('data')
X_train, y_train = mndata.load_training()
X_test, y_test = mndata.load_testing()

# convert data to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)


#%% Data Preprocessing

# Data Deskewing
for i, img in enumerate(X_train):
    dimg = deskew(img.reshape(28,28))
    X_train[i] = dimg.reshape(784,)

for i, img in enumerate(X_test):
    dimg = deskew(img.reshape(28,28))
    X_test[i] = dimg.reshape(784,)


# Data Normalization
maxValue = np.amax(X_train)
minValue = np.amin(X_train)

X_train = (X_train - minValue)/(maxValue-minValue)
X_test = (X_test - minValue)/(maxValue-minValue)



#%% Data Processing

clf = MLPClassifier(solver='adam', alpha=1e-5, activation='relu', max_iter=100000, hidden_layer_sizes=(600))
# clf = SVC(kernel='poly', degree=4, tol=1e-5)

# estimators = [
#     ('rf', RandomForestClassifier(n_estimators=100, max_depth=17)),
#     ('svm', SVC(kernel='poly', degree=4, tol=1e-5)),
#     ('nn',MLPClassifier(solver='adam', alpha=1e-5, activation='relu', max_iter=100000, hidden_layer_sizes=(600)))
#      ]
#
# clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

clf.fit(X_train, y_train)


#%% Model Evaluation

y_predict = np.array(clf.predict(X_test))
true_predict = 0 
for i, j in zip(y_test, y_predict):
    if i == j:
        true_predict += 1
        
accuracy = true_predict / len(y_test)


print('Overall Accuracy: {} \n'.format(accuracy))
print("Accuracy per class:")

for digit in np.unique(y_test):
    idxs = np.where(y_test == digit)
    yt = y_test[idxs]
    yp = y_predict[idxs]
    
    true_predict = 0 
    for i, j in zip(yt, yp):
        if i == j:
            true_predict += 1            
    
    accuracy = true_predict / len(idxs[0])        
    print("Class {}: {}".format(digit, round(accuracy,4)))
    
#%% Save Model
print("\nModel saved")
clf = dump(clf, 'classification-model.joblib') 
