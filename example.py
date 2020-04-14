# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 15:08:39 2020

@author: Angelo
"""

# import function
import digits_recognition

# define images to be processed
input_data = ['example-data/0.jpg',
              'example-data/1.jpg',
              'example-data/2.jpg',
              'example-data/3.jpg',
              'example-data/4.jpg',
              'example-data/5.jpg',
              'example-data/6.jpg',
              'example-data/7.jpg',
              'example-data/8.jpg',
              'example-data/9.jpg']

predictions = digits_recognition.recognizeDigits(input_data)

for i, p in enumerate(predictions):
    filename = input_data[i].split('/')[1]
    print('{} file was recognized as: {}'.format(filename,p))