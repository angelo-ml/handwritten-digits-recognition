# -*- coding: utf-8 -*-
"""
========================================
== Hand-written digits classification ==
========================================
Created on Thu Apr  2 22:48:39 2020
@author: Malandrakis Angelos
"""


#%% Import Libraries & Data

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import interpolation
from joblib import load

#%% Load Classification model
clf = load('classification-model.joblib')


#%% Define Deskewing functions
def _moments(image):
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
    c,v = _moments(image)
    alpha = v[0,1]/v[0,0]
    affine = np.array([[1,0],[alpha,1]])
    ocenter = np.array(image.shape)/2.0
    offset = c-np.dot(affine,ocenter)
    return interpolation.affine_transform(image,affine,offset=offset)


#%% digits recognintion function (input: list with images' paths)
def recognizeDigits(img_path_list):
    
    # prepare input data array
    total_imgs = len(img_path_list)
    input_imgs = np.zeros([total_imgs,784])
    
    # input data preprocessing
    for i, img_path in enumerate(img_path_list):
        img = plt.imread(img_path) # read image
        new_img = np.mean(img,2) # make image black & white
        new_img = np.abs(255-new_img) # inverse image colors
        
        # deskew image
        new_img = deskew(new_img)
        
        # normalize image values
        maxValue = np.amax(new_img)
        minValue = np.amin(new_img)
        new_img = (new_img - minValue)/(maxValue-minValue)
        
        # rehape image and insert to input data array
        reshaped_img = new_img.reshape(784,)
        input_imgs[i] = reshaped_img
        
    return(clf.predict(input_imgs))