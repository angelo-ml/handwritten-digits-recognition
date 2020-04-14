# Python Handwritten Digits Recognition
Handwritten Digits Recognition is a python class which recognizes images of handwritten digits with size 28x28 pixels with 98.8% accuracy.

## Repository Files
- predicting_hand_written_digits: trains a neurall network to recognize digits out of 28x28px images taken from the MNIST database (http://yann.lecun.com/exdb/mnist/)
- classification-model.joblib: Is the final classification model
- digits_recognition: Contains methods *deskew* and *recognizeDigits* that deskew the input images and recognize them as digits between 0-9.
- example: is an example implementation of the class, using an image of my handritten digits.

## Usage
**digits_recognition.recognizeDigits()** method, receives as input data a python list with the images paths and returns a lists of integers with the digit recognized in each image.

```
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

# list with the digits recognized in each image
predictions = digits_recognition.recognizeDigits(input_data)
```

**digits_recognition.deskew()** method, is used in the preprocessing part of the *recognizeDigits()* method. However, it can also be used independently to deskew an image.